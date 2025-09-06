import * as faceapi from 'face-api.js';
import { useCallback, useEffect, useRef, useState } from 'react';

// --- OpenCV.js loader (module-level) ---
let __opencvReady = false as boolean;
let __opencvPromise: Promise<void> | null = null;
function ensureOpenCV(opencvUrl = '/opencv/opencv.js'): Promise<void> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (__opencvReady && (window as any).cv) return Promise.resolve();
    if (__opencvPromise) return __opencvPromise;
    __opencvPromise = new Promise<void>((resolve, reject) => {
        const done = () => {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const cv: any = (window as any).cv;
            if (!cv) {
                reject(new Error('cv not present after load'));
                return;
            }
            if (cv.getBuildInformation) {
                __opencvReady = true;
                resolve();
                return;
            }
            cv.onRuntimeInitialized = () => {
                __opencvReady = true;
                resolve();
            };
        };
        const existing = document.querySelector<HTMLScriptElement>(`script[src="${opencvUrl}"]`);
        if (existing) {
            done();
            return;
        }
        const s = document.createElement('script');
        s.src = opencvUrl;
        s.async = true;
        s.onload = done;
        s.onerror = () => reject(new Error('Failed to load OpenCV.js'));
        document.head.appendChild(s);
    });
    return __opencvPromise;
}
function isOpenCVReady() {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return __opencvReady && Boolean((window as any).cv);
}

// Mobile-first, fullscreen selfie capture with overlay selector (Face/Card)
// + face-api.js integration to detect when a face is inside the guide (stroke turns light blue)
// + heuristic card detection (edge strength) as a lightweight alternative to OpenCV.js
// - 1440x2560 portrait capture (final image has NO overlay)
// - Preview is mirrored like a selfie; final capture is NOT mirrored

export type OverlayMode = 'face' | 'card';

interface Box {
    x: number;
    y: number;
    width: number;
    height: number;
}

interface Mapping {
    cW: number;
    cH: number;
    vw: number;
    vh: number;
    scale: number;
    dx: number;
    dy: number;
    mirrored: boolean;
}

interface LightIdCheckProps {
    // URL path to face-api models (relative to site root)
    faceModelsUrl?: string;
    // URL path to OpenCV.js (relative to site root)
    opencvUrl?: string;

    //Open Overlay
    isOpen?: boolean;

    // Initial overlay mode
    initialOverlay?: OverlayMode;

    // Callback when capture is taken (dataUrl is JPEG base64)
    onCapture: (dataUrl: string | null) => void;

    //Automatic capture after detection (for testing/demo)
    autoCapture?: boolean;
    autoCaptureDelayMs?: number;

    // Optional className for the outer container
    className?: string;
}

export function LightIdCheck({
    faceModelsUrl = '/models',
    opencvUrl = '/opencv/opencv.js',
    isOpen = false,
    initialOverlay = 'face',
    onCapture,
    autoCapture = false,
    autoCaptureDelayMs = 5000,
    className,
}: LightIdCheckProps) {
    // State
    const [overlay, setOverlay] = useState<OverlayMode>(initialOverlay);
    const overlayRef = useRef<OverlayMode>(initialOverlay);
    const [error, setError] = useState<string | null>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [capturePending, setCapturePending] = useState(false);

    const videoRef = useRef<HTMLVideoElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const rafRef = useRef<number | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    // face-api + detection
    const faceModelsLoadedRef = useRef(false);
    const detectionTimerRef = useRef<number | null>(null);
    const lastFaceBoxRef = useRef<Box | null>(null);
    const faceInsideRef = useRef(false);
    const cardOkRef = useRef(false);
    const mappingRef = useRef<Mapping | null>(null);
    // (offscreen canvas handled as a module-level singleton; see getOffscreen())

    useEffect(() => {
        overlayRef.current = overlay;
    }, [overlay]);

    const stopStream = useCallback(() => {
        if (rafRef.current) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
        }
        if (detectionTimerRef.current) {
            window.clearInterval(detectionTimerRef.current);
            detectionTimerRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop());
            streamRef.current = null;
        }
        setIsStreaming(false);
    }, []);

    const close = useCallback(() => {
        stopStream();
        if (onCapture) onCapture(null); // signal closed without capture
    }, [onCapture, stopStream]);

    const ensureFaceModels = useCallback(async (modelsUrl = '/models') => {
        if (faceModelsLoadedRef.current) return;
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(modelsUrl),
            // Optionally: landmarks if you want finer checks
            // faceapi.nets.faceLandmark68TinyNet.loadFromUri(modelsUrl),
        ]);
        faceModelsLoadedRef.current = true;
    }, []);

    // === DRAW LOOP: render video + overlay (color reflects detection state) ===
    const draw = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const mirrored = overlayRef.current === 'face';
        if (!video || !canvas) {
            rafRef.current = requestAnimationFrame(draw);
            return;
        }
        if (video.readyState < 2) {
            rafRef.current = requestAnimationFrame(draw);
            return;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            rafRef.current = requestAnimationFrame(draw);
            return;
        }

        // Fullscreen portrait sizing
        const targetW = canvas.clientWidth || window.innerWidth || 1440;
        const targetH = canvas.clientHeight || window.innerHeight || 2560;
        const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
        const cW = Math.floor(targetW * dpr),
            cH = Math.floor(targetH * dpr);
        if (canvas.width !== cW || canvas.height !== cH) {
            canvas.width = cW;
            canvas.height = cH;
        }

        const vw = video.videoWidth || 1440;
        const vh = video.videoHeight || 2560;
        const scale = Math.max(cW / vw, cH / vh);
        const dw = vw * scale,
            dh = vh * scale;
        const dx = Math.floor((cW - dw) / 2),
            dy = Math.floor((cH - dh) / 2);

        // share mapping for detection loop
        mappingRef.current = { cW, cH, vw, vh, scale, dx, dy, mirrored };

        // Clear + draw mirrored video (selfie)
        ctx.clearRect(0, 0, cW, cH);
        if (mirrored) {
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(video, -dx - dw, dy, dw, dh);
            ctx.restore();
        } else {
            ctx.drawImage(video, dx, dy, dw, dh);
        }

        // === Overlay ===
        const current = overlayRef.current;

        // Dim whole screen
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.fillRect(0, 0, cW, cH);

        // Build path for the guide
        let guidePath: Path2D;
        if (current === 'face') {
            const cx = cW / 2;
            const cy = cH * 0.5;
            const w = Math.min(cW, cH) * 0.7;
            const h = w * 1.25;
            guidePath = faceSilhouettePath2D(cx, cy, w, h);
        } else {
            const { x, y, w, h } = cardOverlayRectCanvas(cW, cH);
            guidePath = roundedRectPath2D(x, y, w, h, Math.min(w, h) * 0.06);
        }

        // Cutout
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fill(guidePath);

        // Border color based on detection result
        const ok =
            current === 'face' ? faceInsideRef.current : cardOkRef.current;
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = ok
            ? 'rgba(125, 211, 252, 1)'
            : 'rgba(255,255,255,0.95)'; // light blue when OK
        ctx.lineWidth = 3 * dpr;
        ctx.stroke(guidePath);

        ctx.restore();

        rafRef.current = requestAnimationFrame(draw);
    }, []);

    // Capture WITHOUT overlay at 1440x2560 portrait (not mirrored)
    const capture = useCallback(() => {
        const video = videoRef.current;
        if (!video) return;
        const outW = 1440,
            outH = 2560;
        const off = document.createElement('canvas');
        off.width = outW;
        off.height = outH;
        const ctx = off.getContext('2d');
        if (!ctx) return;

        const vw = video.videoWidth || outW;
        const vh = video.videoHeight || outH;
        const scale = Math.max(outW / vw, outH / vh);
        const dw = vw * scale,
            dh = vh * scale;
        const dx = (outW - dw) / 2,
            dy = (outH - dh) / 2;
        ctx.drawImage(video, dx, dy, dw, dh);

        const dataUrl = off.toDataURL('image/jpeg', 0.95);
        if (onCapture) onCapture(dataUrl);

        // For demo purposes, open in new window
        // In real use, you would probably want to upload the dataUrl to a server
        // const win = window.open();
        // if (win)
        //     win.document.write(
        //         `<img src="${dataUrl}" style="width:100%;height:auto"/>`
        //     );
    }, [onCapture]);

    // === DETECTION LOOP ===
    const detectionTick = useCallback(async () => {
        const video = videoRef.current;
        const map = mappingRef.current;
        if (!video || !map) return;

        if (overlayRef.current === 'face' && faceModelsLoadedRef.current) {
            try {
                const det = await faceapi.detectSingleFace(
                    video,
                    new faceapi.TinyFaceDetectorOptions({
                        inputSize: 224,
                        scoreThreshold: 0.5,
                    })
                );
                lastFaceBoxRef.current = det?.box
                    ? {
                        x: det.box.x,
                        y: det.box.y,
                        width: det.box.width,
                        height: det.box.height,
                    }
                    : null;
                faceInsideRef.current = false;
                if (lastFaceBoxRef.current) {
                    const { cW, cH, vw, scale, dx, dy } = map;
                    const b = lastFaceBoxRef.current;
                    // map video -> canvas (mirrored)
                    const left = dx + (vw - (b.x + b.width)) * scale;
                    const top = dy + b.y * scale;
                    const centerX = left + (b.width * scale) / 2;
                    const centerY = top + (b.height * scale) / 2;

                    const p = faceSilhouettePath2D(
                        cW / 2,
                        cH * 0.5,
                        Math.min(cW, cH) * 0.7,
                        Math.min(cW, cH) * 0.5 * 1.25
                    );
                    const ctx = canvasRef.current?.getContext('2d');
                    if (ctx) {
                        faceInsideRef.current = ctx.isPointInPath(
                            p,
                            centerX,
                            centerY
                        );
                    }
                }
            } catch {
                /* ignore transient errors */
            }
        } else if (overlayRef.current === 'card') {
            try {
                const { cW, cH, vw, vh, scale, dx, dy, mirrored } = map;
                const rc = cardOverlayRectCanvas(cW, cH);
                const rv = canvasRectToVideoRect(rc, {
                    vw,
                    vh,
                    scale,
                    dx,
                    dy,
                    mirrored,
                });
                try {
                    await ensureOpenCV(opencvUrl);
                    if (isOpenCVReady()) {
                        cardOkRef.current = await estimateCardOpenCV(video, rv);
                    } else {
                        cardOkRef.current = await estimateCardHeuristic(
                            video,
                            rv
                        );
                    }
                } catch {
                    cardOkRef.current = await estimateCardHeuristic(video, rv);
                }
            } catch {
                /* ignore */
            }
        }

        // Auto-capture if enabled and detection is OK
        if (autoCapture && !capturePending) {
            const ok =
                overlayRef.current === 'face'
                    ? faceInsideRef.current
                    : cardOkRef.current;
            if (ok) {
                setCapturePending(true);
                setTimeout(() => {
                    if (overlayRef.current === 'face'
                        ? faceInsideRef.current
                        : cardOkRef.current) {
                        capture();
                    }
                    setCapturePending(false);
                }, autoCaptureDelayMs);
            }
        }
    }, [autoCapture, autoCaptureDelayMs, opencvUrl]);

    const start = useCallback(async () => {
        setError(null);
        try {
            // Load face models in parallel; don't block UI if card mode is selected first
            ensureFaceModels(faceModelsUrl).catch(() => {
                /* non-fatal */
            });

            const constraints: MediaStreamConstraints = {
                video: {
                    facingMode: { ideal: 'user' },
                    width: { ideal: 1440 },
                    height: { ideal: 2560 },
                    aspectRatio: 9 / 16,
                },
                audio: false,
            };
            const stream = await navigator.mediaDevices.getUserMedia(
                constraints
            );
            streamRef.current = stream;
            const video = videoRef.current;
            if (!video) return;
            video.srcObject = stream;
            video.playsInline = true; // iOS/Safari

            const onPlaying = () => {
                setIsStreaming(true);
                if (!rafRef.current)
                    rafRef.current = requestAnimationFrame(draw);
                if (!detectionTimerRef.current)
                    detectionTimerRef.current = window.setInterval(
                        detectionTick,
                        333
                    );
            };
            video.addEventListener('playing', onPlaying, { once: true });
            await video.play();
        } catch (e: unknown) {
            setError(
                (e as Error)?.message ??
                'Camera access failed. Check permissions.'
            );
            setIsStreaming(false);
        }
    }, [ensureFaceModels, faceModelsUrl]);

    useEffect(() => {
        if (!isOpen) return;
        start();
        return () => stopStream();
    }, [isOpen, start, stopStream]);

    return (
        <div className={"w-full "+ className}>
            {isOpen && (
                <div
                    className="fixed inset-0 z-50 bg-black"
                    style={{
                        paddingTop: 'env(safe-area-inset-top)',
                        paddingBottom: 'env(safe-area-inset-bottom)',
                        paddingLeft: 'env(safe-area-inset-left)',
                        paddingRight: 'env(safe-area-inset-right)',
                    }}
                >
                    {/* Top bar */}
                    <div className="absolute top-0 right-0 left-0 z-10 flex items-center justify-between px-4 py-3 text-white/95">
                        <div className="flex items-center gap-2">
                            <button
                                onClick={close}
                                className="rounded-xl border border-white/20 bg-white/10 px-3 py-1.5 backdrop-blur"
                            >
                                Close
                            </button>
                        </div>
                        <div className="flex items-center gap-2">
                            <Segmented
                                value={overlay}
                                onChange={setOverlay}
                                options={[
                                    { value: 'face', label: 'Face' },
                                    { value: 'card', label: 'Card' },
                                ]}
                            />
                            <button
                                onClick={capture}
                                disabled={!isStreaming}
                                className="rounded-xl border border-white/20 bg-blue-600 px-3 py-1.5 text-white disabled:opacity-50"
                            >
                                Capture
                            </button>
                        </div>
                    </div>

                    {/* Video + Canvas preview (fills screen) */}
                    <div className="absolute inset-0">
                        <video ref={videoRef} className="hidden" />
                        <canvas
                            ref={canvasRef}
                            className="block h-full w-full"
                            style={{ touchAction: 'none' }}
                        />
                        {!isStreaming && (
                            <div className="absolute inset-0 grid place-items-center text-white/90">
                                <div className="text-center">
                                    <div className="text-lg font-semibold">
                                        Starting camera…
                                    </div>
                                    {error && (
                                        <div className="mt-2 text-red-300">
                                            {error}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

function Segmented<T extends string>({
    value,
    onChange,
    options,
}: {
    value: T;
    onChange: (v: T) => void;
    options: { value: T; label: string }[];
}) {
    return (
        <div className="flex overflow-hidden rounded-xl border border-white/20 bg-white/10 backdrop-blur">
            {options.map((opt) => (
                <button
                    key={opt.value}
                    onClick={() => onChange(opt.value)}
                    className={`px-3 py-1.5 text-sm ${value === opt.value ? 'bg-white text-black' : 'text-white/90'}`}
                >
                    {opt.label}
                </button>
            ))}
        </div>
    );
}

// === PATH HELPERS ===
function faceSilhouettePath2D(cx: number, cy: number, w: number, h: number) {
    const p = new Path2D();
    const rx = w / 2,
        ry = h / 1.8;
    const topY = cy - ry;
    const chinY = cy + ry * 0.86;
    const browY = cy - ry * 0.35;
    const cheekY = cy + ry * 0;
    const jawY = cy + ry * 0.55;
    const craniumX = rx * 1.05,
        templeX = rx * 0.95,
        cheekX = rx * 0.9,
        jawX = rx * 0.69;

    p.moveTo(cx, topY);
    p.bezierCurveTo(cx + craniumX, topY, cx + templeX, browY, cx + templeX, browY);
    p.bezierCurveTo(cx + templeX, browY + ry * 0.1, cx + cheekX, cheekY, cx + cheekX, cheekY);
    p.bezierCurveTo(cx + cheekX * 0.95, cheekY + ry * 0.2, cx + jawX, jawY, cx + jawX, jawY);
    p.bezierCurveTo(cx + jawX * 0.85, jawY + ry * 0.2, cx + (w * 0.2) / 2, chinY, cx, chinY);
    p.bezierCurveTo(cx - (w * 0.2) / 2, chinY, cx - jawX * 0.85, jawY + ry * 0.2, cx - jawX, jawY);
    p.bezierCurveTo(cx - jawX, jawY, cx - cheekX * 0.95, cheekY + ry * 0.2, cx - cheekX, cheekY);
    p.bezierCurveTo(cx - cheekX, cheekY, cx - templeX, browY + ry * 0.1, cx - templeX, browY);
    p.bezierCurveTo(cx - templeX, browY, cx - craniumX, topY, cx, topY);
    p.closePath();
    return p;
}

function roundedRectPath2D(x: number, y: number, w: number, h: number, r: number) {
    const p = new Path2D();
    p.moveTo(x + r, y);
    p.lineTo(x + w - r, y);
    p.quadraticCurveTo(x + w, y, x + w, y + r);
    p.lineTo(x + w, y + h - r);
    p.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    p.lineTo(x + r, y + h);
    p.quadraticCurveTo(x, y + h, x, y + h - r);
    p.lineTo(x, y + r);
    p.quadraticCurveTo(x, y, x + r, y);
    p.closePath();
    return p;
}

function cardOverlayRectCanvas(cW: number, cH: number) {
    const w = cW * 0.9;
    const h = w / 1.586;
    const x = (cW - w) / 2;
    const y = cH * 0.55 - h / 2;
    return { x, y, w, h };
}

function canvasRectToVideoRect(
    rc: { x: number; y: number; w: number; h: number },
    m: { vw: number; vh: number; scale: number; dx: number; dy: number; mirrored: boolean },
) {
    const wv = rc.w / m.scale;
    const hv = rc.h / m.scale;
    const xv = m.mirrored
        ? m.vw - (rc.x - m.dx) / m.scale - wv // mirrored case
        : (rc.x - m.dx) / m.scale; // non-mirrored
    const yv = (rc.y - m.dy) / m.scale;
    return { x: Math.max(0, xv), y: Math.max(0, yv), w: Math.min(wv, m.vw), h: Math.min(hv, m.vh) };
}

// === Heuristic card detector (edge strength along rectangle borders) ===
async function estimateCardHeuristic(video: HTMLVideoElement, rv: { x: number; y: number; w: number; h: number }) {
    const off = getOffscreen();
    // Normalize to ~320px width for speed
    const outW = 320;
    const outH = Math.max(1, Math.round(outW / 1.586));
    off.width = outW;
    off.height = outH;
    const ctx = off.getContext('2d');
    if (!ctx) return false;

    // Draw the video ROI (NOT mirrored)
    ctx.drawImage(video, rv.x, rv.y, rv.w, rv.h, 0, 0, outW, outH);
    const img = ctx.getImageData(0, 0, outW, outH);
    const score = edgeScore(img.data, outW, outH);
    // Tunable threshold: higher requires stronger, cleaner edges
    return score > 0.18; // ~18% normalized edge contrast
}

let __OFFSCREEN_CANVAS__: HTMLCanvasElement | null = null;
function getOffscreen() {
    if (!__OFFSCREEN_CANVAS__) __OFFSCREEN_CANVAS__ = document.createElement('canvas');
    return __OFFSCREEN_CANVAS__;
}

function edgeScore(data: Uint8ClampedArray, w: number, h: number) {
    const lum = (i: number) => 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    const idx = (x: number, y: number) => (y * w + x) * 4;
    const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

    const samples = 200;
    let acc = 0;
    let n = 0;
    for (let s = 0; s < samples; s++) {
        const x = 2 + Math.floor(((w - 4) * s) / samples);
        const y1 = 2,
            y2 = Math.min(5, h - 1);
        acc += Math.abs(lum(idx(x, y1)) - lum(idx(x, y2))) / 255;
        n++;
        const yb1 = h - 3,
            yb2 = clamp(h - 6, 0, h - 1);
        acc += Math.abs(lum(idx(x, yb1)) - lum(idx(x, yb2))) / 255;
        n++;
    }
    for (let s = 0; s < samples; s++) {
        const y = 2 + Math.floor(((h - 4) * s) / samples);
        const x1 = 2,
            x2 = Math.min(5, w - 1);
        acc += Math.abs(lum(idx(x1, y)) - lum(idx(x2, y))) / 255;
        n++;
        const xr1 = w - 3,
            xr2 = clamp(w - 6, 0, w - 1);
        acc += Math.abs(lum(idx(xr1, y)) - lum(idx(xr2, y))) / 255;
        n++;
    }
    return acc / n; // 0..1
}

// === OpenCV.js card detector ===
async function estimateCardOpenCV(video: HTMLVideoElement, rv: { x: number; y: number; w: number; h: number }) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const cv: any = (window as any).cv;
    if (!cv) return false;
    const off = getOffscreen();
    const outW = 540;
    const outH = Math.max(1, Math.round(outW / 1.586));
    off.width = outW;
    off.height = outH;
    const ctx2d = off.getContext('2d');
    if (!ctx2d) return false;
    // Crop ROI from the native (non-mirrored) video
    ctx2d.drawImage(video, rv.x, rv.y, rv.w, rv.h, 0, 0, outW, outH);

    const src = cv.imread(off);
    const gray = new cv.Mat();
    const blur = new cv.Mat();
    const edges = new cv.Mat();
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));

    // Preprocess
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.equalizeHist(gray, gray);
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);

    // Auto-Canny thresholds using mean as a robust proxy
    const data = blur.data as Uint8Array;
    let sum = 0;
    data.forEach((v) => (sum += v));
    const mean = sum / data.length;
    const lower = Math.max(10, mean * 0.66);
    const upper = Math.min(255, mean * 1.33);
    cv.Canny(blur, edges, lower, upper);
    cv.dilate(edges, edges, kernel);

    // Contours
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const areaROI = outW * outH;
    let ok = false;

    for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        const area = cv.contourArea(cnt);
        if (area < areaROI * 0.05) {
            cnt.delete();
            continue;
        }

        // Rotated rectangle fit
        const rr = cv.minAreaRect(cnt);
        const rectArea = rr.size.width * rr.size.height;
        if (rectArea <= 0) {
            cnt.delete();
            continue;
        }

        const major = Math.max(rr.size.width, rr.size.height);
        const minor = Math.max(1, Math.min(rr.size.width, rr.size.height));
        const ar = major / minor; // rotation-invariant aspect ratio

        const rectangularity = area / rectArea; // 0..1 (1 is perfect fill)

        if (rectArea > areaROI * 0.12 && rectArea < areaROI * 0.98 && ar > 1.35 && ar < 1.9 && rectangularity > 0.6) {
            ok = true;
            cnt.delete();
            break;
        }

        cnt.delete();
    }

    // Cleanup
    src.delete();
    gray.delete();
    blur.delete();
    edges.delete();
    contours.delete();
    hierarchy.delete();
    kernel.delete();
    return ok;
}
