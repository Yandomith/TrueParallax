import * as THREE from 'three';
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

// --- Configuration ---
const IS_MOBILE = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const VIDEO_WIDTH = IS_MOBILE ? 320 : 640;
const VIDEO_HEIGHT = IS_MOBILE ? 240 : 480;
const PARALLAX_FACTOR = 0.8; // Sensitivity of the effect
const DEPTH_MULTIPLIER = 0.05; // How strongly distance amplifies parallax
// Base max rotation (radians) when cube scale.z === 1. At `scale.z === 1`
// this will allow +/-10 degrees; the allowed range scales proportionally
// with the cube's Z size (e.g., scale.z === 100 -> +/-1000 degrees).
const BASE_MAX_ANCHOR_ROTATION = THREE.MathUtils.degToRad(10);
const HEAD_ROTATION_FACTOR = THREE.MathUtils.degToRad(30);
const MAX_PIXEL_RATIO = IS_MOBILE ? 1 : 1.5;
const FACE_DETECTION_INTERVAL_MS = IS_MOBILE ? 100 : 66;
const DEBUG_UI_INTERVAL_MS = IS_MOBILE ? 250 : 100;

// --- Globals ---
let scene, camera, renderer;
let faceLandmarker;
let lastVideoTime = -1;
let lastFaceDetectionTime = 0;
let lastDebugUpdateTime = 0;
let video;
let cube;
let pivotDot;
let cubeAnchor;
let childCube;
let childCubes = [];
const CHILD_GEO_SIZE = 1;
// Navigation / camera smoothing
let targetCameraPos;
let headRotationTarget = new THREE.Vector2(0, 0);
const CAMERA_LERP = 0.15;
const scratchWorldPos = new THREE.Vector3();
const scratchWorldQuat = new THREE.Quaternion();
const scratchWorldEuler = new THREE.Euler();
const scratchChildWorldPos = new THREE.Vector3();
const scratchChildWorldScale = new THREE.Vector3();
const scratchChildWorldQuat = new THREE.Quaternion();
const scratchChildWorldEuler = new THREE.Euler();
const CAMERA_OPTIONS = {
    antialias: !IS_MOBILE,
    alpha: false,
    powerPreference: 'high-performance',
    precision: IS_MOBILE ? 'mediump' : 'highp'
};

// --- Initialization ---
async function init() {
    // 1. Setup Three.js
    setupThreeJS();

    // Start rendering immediately while the heavier resources load in the background.
    animate();

    // 1b. Load HDRI environment lighting without blocking first paint.
    void setupHDRI().catch((error) => {
        console.warn('HDRI load failed:', error);
    });

    // 2. Setup MediaPipe in the background.
    void setupMediaPipe().catch((error) => {
        console.warn('MediaPipe setup failed:', error);
    });

    // 3. Setup Webcam in the background.
    void setupWebcam().catch((error) => {
        console.warn('Webcam setup failed:', error);
    });

    // navigation gizmo removed
}

function setupThreeJS() {
    const container = document.getElementById('container');

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    // Camera
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.position.set(0, 0, 5); // Initial position
    // Initialize target camera position for smooth navigation
    targetCameraPos = camera.position.clone();

    // Renderer
    renderer = new THREE.WebGLRenderer(CAMERA_OPTIONS);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    container.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 2, 100);
    pointLight.position.set(0, 5, 5);
    scene.add(pointLight);

    const roomGeometry = new THREE.BoxGeometry(30, 20, 100);
    const roomMaterial = new THREE.MeshPhysicalMaterial({
        color: 0x222222,
        side: THREE.BackSide, // Render inside of the box
        roughness: 0.5,
        metalness: 0.1,
        clearcoat: 0.0,
        reflectivity: 0.5
    });
    const room = new THREE.Mesh(roomGeometry, roomMaterial);
    scene.add(room);

    // --- Floating Object ---
    const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);

    cubeAnchor = new THREE.Group();
    cubeAnchor.position.set(0, 0, 0);
    scene.add(cubeAnchor);

    // visible from inside/outside. Keep the pivot-facing face transparent.
    const cubeMaterials = [
        new THREE.MeshPhysicalMaterial({ color: 0xff3355, side: THREE.DoubleSide, roughness: 0.4, metalness: 0.05 }), // +X
        new THREE.MeshPhysicalMaterial({ color: 0xff8888, side: THREE.DoubleSide, roughness: 0.4, metalness: 0.05 }), // -X
        new THREE.MeshPhysicalMaterial({ color: 0xff22aa, side: THREE.DoubleSide, roughness: 0.4, metalness: 0.05 }), // +Y
        new THREE.MeshPhysicalMaterial({ color: 0x33ccff, side: THREE.DoubleSide, roughness: 0.4, metalness: 0.05 }), // -Y
        // Hide the face that sits on the pivot side.
        new THREE.MeshPhysicalMaterial({ color: 0x3333ff, transparent: true, opacity: 0, side: THREE.DoubleSide }), // +Z (pivot-facing)
        new THREE.MeshPhysicalMaterial({ color: 0xffcc33, side: THREE.DoubleSide, roughness: 0.35, metalness: 0.05 })  // -Z
    ];

    cube = new THREE.Mesh(cubeGeometry, cubeMaterials);
    // Default Z depth large by user request; scale from center
    cube.scale.set(1, 1, 100);
    // Place cube center at world origin so rotations happen in-place
    cube.position.set(0, 0, 0);
    cube.userData.initialPos = { x: 0, y: 0, z: 0 };
    cubeAnchor.add(cube);

    // Small visible marker for the pivot point.
    pivotDot = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xffff00 })
    );
    pivotDot.position.set(0, 0, 0);
    cubeAnchor.add(pivotDot);

    // Create the child cube and initialize the Z slider UI (helpers below).
    createChildCube();
    initZSlider();
    initChildControls();


    // Handle resize
    window.addEventListener('resize', onWindowResize);

    // Make initial fit to screen
    fitCubeToScreen();
}

async function setupHDRI() {
    if (!renderer || !scene) return;

    const loader = new RGBELoader();
    const hdrTexture = await loader.loadAsync(
        new URL('./golden_gate_hills_1k.hdr', import.meta.url).href
    );

    hdrTexture.mapping = THREE.EquirectangularReflectionMapping;

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    pmremGenerator.compileEquirectangularShader();

    const envMap = pmremGenerator.fromEquirectangular(hdrTexture).texture;
    scene.environment = envMap;
    scene.background = new THREE.Color(0x111111);

    hdrTexture.dispose();
    pmremGenerator.dispose();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    // no gizmo renderer

    // Refit cube so its front face fills the screen
    fitCubeToScreen();
}

// Scale the cube X/Y so its front face fills the camera viewport at the
// cube's front-face distance (front face is positioned at world z == 0).
function fitCubeToScreen() {
    if (!camera || !cube) return;

    // Distance from camera to origin (front face sits at z==0)
    const distance = camera.position.distanceTo(new THREE.Vector3(0, 0, 0));

    // Vertical field of view in radians
    const vFOV = THREE.MathUtils.degToRad(camera.fov);
    const visibleHeight = 2 * distance * Math.tan(vFOV / 2);
    const visibleWidth = visibleHeight * camera.aspect;

    // Since the cube geometry is 1 unit wide/high, scale by the visible size.
    // Keep the cube centered at origin so scaling is symmetric about (0,0,0).
    cube.scale.x = visibleWidth;
    cube.scale.y = visibleHeight;
    // Do not modify cube.position — keep center at origin
    cube.userData.initialPos.z = cube.position.z;
    neutralizeChildScale();
}

// Create a small child cube attached to `cube` at the top-left-front in local
// cube space. The child is added as a child so it inherits position/rotation.
function createChildCube(pos = new THREE.Vector3(0, 0, 0)) {
    if (!cube) return;
    const childGeo = new THREE.BoxGeometry(CHILD_GEO_SIZE, CHILD_GEO_SIZE, CHILD_GEO_SIZE);
    const childMat = new THREE.MeshPhysicalMaterial({ color: 0x00ff00, side: THREE.DoubleSide, roughness: 0.4, metalness: 0.05 });
    const mesh = new THREE.Mesh(childGeo, childMat);
    // Place the child inside the parent cube (local coordinates).
    mesh.position.copy(pos);
    mesh.name = `child_${childCubes.length}`;
    cube.add(mesh);
    childCubes.push(mesh);
    // Keep `childCube` as a reference to the first child for compatibility.
    if (!childCube) childCube = mesh;
    updateChildCountUI();
    return mesh;
}

// Initialize the Z-size slider UI and bind it to `cube.scale.z`.
function initZSlider() {
    const zSlider = document.getElementById('z-size-slider');
    const zValueDisplay = document.getElementById('z-size-value');
    if (!zSlider || !zValueDisplay || !cube) return;
    zSlider.value = (cube.scale.z || 100).toString();
    zValueDisplay.innerText = parseFloat(zSlider.value).toFixed(2);
    zSlider.addEventListener('input', (ev) => {
        const v = parseFloat(ev.target.value);
        cube.scale.z = v;
        // Keep scaling centered at origin (no position change)
        cube.userData.initialPos.z = cube.position.z;
        zValueDisplay.innerText = v.toFixed(2);
        neutralizeChildScale();
    });
}

// Prevent the child from inheriting the parent's scale so it keeps a
// constant world size while still inheriting rotation/position.
function neutralizeChildScale() {
    if (!cube || childCubes.length === 0) return;
    const ps = cube.scale;
    const sx = ps.x !== 0 ? 1 / ps.x : 1;
    const sy = ps.y !== 0 ? 1 / ps.y : 1;
    const sz = ps.z !== 0 ? 1 / ps.z : 1;
    for (const c of childCubes) {
        c.scale.set(sx, sy, sz);
    }
}

// Update the on-screen debug readouts for cube and child transforms.
function updateDebugUI() {
    try {
        const now = performance.now();
        if (now - lastDebugUpdateTime < DEBUG_UI_INTERVAL_MS) return;
        lastDebugUpdateTime = now;

        const cubePosEl = document.getElementById('cube-pos');
        const cubeScaleEl = document.getElementById('cube-scale');
        const cubeRotEl = document.getElementById('cube-rot');
        const childPosEl = document.getElementById('child-cube-pos');
        const childScaleEl = document.getElementById('child-cube-scale');
        const childRotEl = document.getElementById('child-cube-rot');
        if (cube) {
            cube.getWorldPosition(scratchWorldPos);
            const s = cube.scale;
            cube.getWorldQuaternion(scratchWorldQuat);
            scratchWorldEuler.setFromQuaternion(scratchWorldQuat, 'XYZ');

            if (cubePosEl) cubePosEl.innerText = `${scratchWorldPos.x.toFixed(2)}, ${scratchWorldPos.y.toFixed(2)}, ${scratchWorldPos.z.toFixed(2)}`;
            if (cubeScaleEl) cubeScaleEl.innerText = `${s.x.toFixed(2)}, ${s.y.toFixed(2)}, ${s.z.toFixed(2)}`;
            if (cubeRotEl) cubeRotEl.innerText = `${THREE.MathUtils.radToDeg(scratchWorldEuler.x).toFixed(1)}°, ${THREE.MathUtils.radToDeg(scratchWorldEuler.y).toFixed(1)}°, ${THREE.MathUtils.radToDeg(scratchWorldEuler.z).toFixed(1)}°`;

            // Show info for the first child (if any) and the total count.
            const countEl = document.getElementById('child-count');
            if (countEl) countEl.innerText = `Children: ${childCubes.length}`;
            const first = childCubes.length > 0 ? childCubes[ 0 ] : null;
            if (first) {
                first.getWorldPosition(scratchChildWorldPos);
                first.getWorldScale(scratchChildWorldScale);
                first.getWorldQuaternion(scratchChildWorldQuat);
                scratchChildWorldEuler.setFromQuaternion(scratchChildWorldQuat, 'XYZ');

                if (childPosEl) childPosEl.innerText = `${scratchChildWorldPos.x.toFixed(2)}, ${scratchChildWorldPos.y.toFixed(2)}, ${scratchChildWorldPos.z.toFixed(2)}`;
                if (childScaleEl) childScaleEl.innerText = `${scratchChildWorldScale.x.toFixed(2)}, ${scratchChildWorldScale.y.toFixed(2)}, ${scratchChildWorldScale.z.toFixed(2)}`;
                if (childRotEl) childRotEl.innerText = `${THREE.MathUtils.radToDeg(scratchChildWorldEuler.x).toFixed(1)}°, ${THREE.MathUtils.radToDeg(scratchChildWorldEuler.y).toFixed(1)}°, ${THREE.MathUtils.radToDeg(scratchChildWorldEuler.z).toFixed(1)}°`;
            }
        }
    } catch (e) {
        // ignore DOM update errors in environments without a document
    }
}

async function setupMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: false,
        runningMode: "VIDEO",
        numFaces: 1
    });
}

// Initialize child controls (button to add more children)
function initChildControls() {
    const addBtn = document.getElementById('add-child-btn');
    if (!addBtn || !cube) return;
    addBtn.addEventListener('click', () => {
        // Place new child near the center with a small random offset (local space)
        const offs = new THREE.Vector3(
            -0.3 + Math.random() * 0.6,
            -0.2 + Math.random() * 0.4,
            -0.001 + Math.random() * -0.35
        );
        createChildCube(offs);
    });
}

function updateChildCountUI() {
    const countEl = document.getElementById('child-count');
    if (countEl) countEl.innerText = `Children: ${childCubes.length}`;
}

async function setupWebcam() {
    video = document.getElementById('webcam');
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: VIDEO_WIDTH },
            height: { ideal: VIDEO_HEIGHT },
            facingMode: 'user',
            frameRate: IS_MOBILE ? { ideal: 15, max: 24 } : { ideal: 24, max: 30 }
        },
        audio: false
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve();
        };
    });
}

// --- Main Loop ---
function animate() {
    requestAnimationFrame(animate);

    // Process Webcam
    const now = performance.now();
    if (faceLandmarker && video && video.currentTime !== lastVideoTime && (now - lastFaceDetectionTime) >= FACE_DETECTION_INTERVAL_MS) {
        lastVideoTime = video.currentTime;
        lastFaceDetectionTime = now;
        const results = faceLandmarker.detectForVideo(video, now);

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[ 0 ];
            updateHeadPosition(landmarks);
        }
    }

    // Render
    // Smoothly move camera toward target (for gizmo navigation)
    if (targetCameraPos) {
        camera.position.lerp(targetCameraPos, CAMERA_LERP);
        camera.lookAt(0, 0, 0);
    }

    // Rotate the cube toward the tracked head position, but keep the
    // allowed rotation capped by the cube's current Z size.
    if (cubeAnchor) {
        const zScale = cube ? Math.abs(cube.scale.z) : 1;
        const maxRotationForThisZ = BASE_MAX_ANCHOR_ROTATION * (zScale / 100);

        const clampedTargetX = THREE.MathUtils.clamp(
            headRotationTarget.x,
            -maxRotationForThisZ,
            maxRotationForThisZ
        );
        const clampedTargetY = THREE.MathUtils.clamp(
            headRotationTarget.y,
            -maxRotationForThisZ,
            maxRotationForThisZ
        );

        cubeAnchor.rotation.x = THREE.MathUtils.lerp(cubeAnchor.rotation.x, clampedTargetX, 0.15);
        cubeAnchor.rotation.y = THREE.MathUtils.lerp(cubeAnchor.rotation.y, clampedTargetY, 0.15);
        cubeAnchor.rotation.z = 0;
    }

    renderer.render(scene, camera);

    // no gizmo rendering

    updateDebugUI();
}

// navigation controls removed

function updateHeadPosition(landmarks) {
    // We use the nose tip (index 1) for X/Y
    const nose = landmarks[ 1 ];

    // Map normalized coordinates to 3D world coordinates
    const x = (nose.x - 0.5) * PARALLAX_FACTOR * 10;
    const y = (0.5 - nose.y) * PARALLAX_FACTOR * 10;

    // Use normalized head position to drive the parent cube rotation.
    // Left/right head movement yaws the cube; up/down movement tilts it.
    headRotationTarget.x = THREE.MathUtils.clamp((0.5 - nose.y) * HEAD_ROTATION_FACTOR, -THREE.MathUtils.degToRad(20), THREE.MathUtils.degToRad(20));
    headRotationTarget.y = THREE.MathUtils.clamp((0.5 - nose.x) * HEAD_ROTATION_FACTOR, -THREE.MathUtils.degToRad(20), THREE.MathUtils.degToRad(20));

    // Lock the single cube's transform so positions do not change.
    if (cube) {
        const initialPos = cube.userData.initialPos;

        cube.position.x = initialPos.x;
        cube.position.y = initialPos.y;
        cube.position.z = initialPos.z;
    }

    // Update Debug Info
    document.getElementById('head-pos').innerText =
        `X: ${x.toFixed(2)}, Y: ${y.toFixed(2)}`;
}

// Start
init();
