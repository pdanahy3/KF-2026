import os
import time
import argparse
import random
from glob import glob
from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from sklearn.neighbors import NearestNeighbors


# ============================================================
# DEFAULTS (mirrors DST.py style)
# ============================================================
IMG_WIDTH = 256
IMG_HEIGHT = 256
STYLE_BATCH_SIZE = 64

STYLE_FOLDER = "./inputs/Gemini_Generated_Image_am1379am1379am13/quadtree2"
MODEL_PATH = "./checkpoints/saved_model.keras"
ONNX_PATH = "./checkpoints/encoder.onnx"

USE_GLOBAL_AVG_POOL_FOR_CODES = False
FILTER_SMALL_STYLE_FILES = False
MIN_STYLE_FILE_SIZE_BYTES = 1200

USE_CANNY_PREPROCESS = False

STYLE_CACHE_SIZE = 128
NN_ALGORITHM = "auto"
NN_METRIC = "euclidean"

SEED = 42
VERBOSE = False


def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)


def rounded_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def _resolve_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, path))


def _restricted_float(min_value: float | None = None, max_value: float | None = None):
    def _convert(v: str) -> float:
        f = float(v)
        if min_value is not None and f < min_value:
            raise argparse.ArgumentTypeError(f"must be >= {min_value}")
        if max_value is not None and f > max_value:
            raise argparse.ArgumentTypeError(f"must be <= {max_value}")
        return f
    return _convert


def _restricted_int(min_value: int | None = None, max_value: int | None = None):
    def _convert(v: str) -> int:
        i = int(v)
        if min_value is not None and i < min_value:
            raise argparse.ArgumentTypeError(f"must be >= {min_value}")
        if max_value is not None and i > max_value:
            raise argparse.ArgumentTypeError(f"must be <= {max_value}")
        return i
    return _convert


def _canny_edges_rgb_float01(img_rgb_uint8: np.ndarray, t1: int = 100, t2: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=int(t1), threshold2=int(t2))
    edges_3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_3.astype(np.float32) / 255.0


def _blend_rgb_with_edges_float01(
    img_rgb_uint8: np.ndarray,
    edge_t1: int = 100,
    edge_t2: int = 200,
    edge_alpha: float = 0.5
) -> np.ndarray:
    rgb = img_rgb_uint8.astype(np.float32) / 255.0
    edges = _canny_edges_rgb_float01(img_rgb_uint8, t1=edge_t1, t2=edge_t2)
    a = float(edge_alpha)
    return (1.0 - a) * rgb + a * edges


def preprocess_for_encoder(img_rgb: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    if img_resized.dtype != np.uint8:
        img_resized = img_resized.astype(np.uint8, copy=False)
    if USE_CANNY_PREPROCESS:
        return _blend_rgb_with_edges_float01(img_resized, edge_t1=100, edge_t2=200, edge_alpha=0.5)
    return img_resized.astype(np.float32) / 255.0


def encode_to_feature_vectors(codes: np.ndarray, use_global_avg_pool: bool = False) -> np.ndarray:
    if codes.ndim == 2:
        return codes.astype(np.float32, copy=False)
    if use_global_avg_pool:
        axes = tuple(range(1, codes.ndim - 1))
        pooled = np.mean(codes, axis=axes, dtype=np.float32)
        return pooled.astype(np.float32, copy=False)
    flat = codes.reshape(codes.shape[0], -1)
    return flat.astype(np.float32, copy=False)


def load_encoder(model_path: str):
    resolved = _resolve_path(model_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Model not found: {resolved}")
    model = load_model(resolved, custom_objects={"rounded_accuracy": rounded_accuracy})
    enc_out = model.layers[-2]
    return tf.keras.models.Model(inputs=model.input, outputs=enc_out.output)


def load_onnx_encoder(onnx_path: str):
    resolved = _resolve_path(onnx_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"ONNX model not found: {resolved}")
    import onnxruntime as ort
    sess = ort.InferenceSession(resolved, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    def _forward(x_nhwc_float32: np.ndarray) -> np.ndarray:
        if x_nhwc_float32.dtype != np.float32:
            x_nhwc_float32 = x_nhwc_float32.astype(np.float32)
        return sess.run([output_name], {input_name: x_nhwc_float32})[0]

    return _forward


class StyleIndex:
    def __init__(
        self,
        encoder_forward,
        style_folder: str,
        img_width: int = 256,
        img_height: int = 256,
        batch_size: int = 32,
        use_global_avg_pool: bool = False,
        filter_small_files: bool = False,
        min_file_size_bytes: int = 0,
    ):
        self.encoder_forward = encoder_forward
        self.style_folder = style_folder
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.use_global_avg_pool = use_global_avg_pool
        self.filter_small_files = filter_small_files
        self.min_file_size_bytes = min_file_size_bytes

    def _gather_files(self):
        files = sorted(glob(os.path.join(self.style_folder, "*")))
        if not files:
            raise FileNotFoundError(f"No style files found in: {self.style_folder}")
        if not self.filter_small_files:
            return files
        kept = []
        for f in files:
            try:
                if os.path.getsize(f) >= self.min_file_size_bytes:
                    kept.append(f)
            except OSError:
                continue
        if not kept:
            raise RuntimeError("All style files were filtered out.")
        return kept

    def build(self):
        print("Building style index...")
        files = self._gather_files()
        codes = []
        valid_paths = []

        batch_imgs = []
        batch_paths = []

        for p in files:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            batch_imgs.append(preprocess_for_encoder(rgb))
            batch_paths.append(p)
            if len(batch_imgs) >= self.batch_size:
                batch = np.stack(batch_imgs, axis=0).astype(np.float32)
                raw = self.encoder_forward(batch)
                vec = encode_to_feature_vectors(raw, use_global_avg_pool=self.use_global_avg_pool)
                codes.append(vec)
                valid_paths.extend(batch_paths)
                batch_imgs, batch_paths = [], []

        if batch_imgs:
            batch = np.stack(batch_imgs, axis=0).astype(np.float32)
            raw = self.encoder_forward(batch)
            vec = encode_to_feature_vectors(raw, use_global_avg_pool=self.use_global_avg_pool)
            codes.append(vec)
            valid_paths.extend(batch_paths)

        if not valid_paths:
            raise RuntimeError("No decodable style images found.")

        style_codes = np.concatenate(codes, axis=0)
        return style_codes, valid_paths


def _open_camera(camera_index: int, width: int | None, height: int | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-frame nearest-neighbor style substitution (no subdivision).")

    p.add_argument("--encoder-backend", choices=["keras", "onnx"], default="keras")
    p.add_argument("--model-path", default=MODEL_PATH, help="Keras .keras path (backend=keras).")
    p.add_argument("--onnx-path", default=ONNX_PATH, help="ONNX encoder path (backend=onnx).")
    p.add_argument("--style-folder", default=STYLE_FOLDER)

    p.add_argument("--img-width", type=_restricted_int(1, None), default=IMG_WIDTH)
    p.add_argument("--img-height", type=_restricted_int(1, None), default=IMG_HEIGHT)
    p.add_argument("--style-batch-size", type=_restricted_int(1, None), default=STYLE_BATCH_SIZE)
    p.add_argument(
        "--use-canny-preprocess",
        action=argparse.BooleanOptionalAction,
        default=USE_CANNY_PREPROCESS,
        help="Overlay Canny edges onto RGB before encoder inference."
    )

    p.add_argument(
        "--use-global-avg-pool-for-codes",
        action=argparse.BooleanOptionalAction,
        default=USE_GLOBAL_AVG_POOL_FOR_CODES
    )
    p.add_argument(
        "--filter-small-style-files",
        action=argparse.BooleanOptionalAction,
        default=FILTER_SMALL_STYLE_FILES
    )
    p.add_argument("--min-style-file-size-bytes", type=_restricted_int(0, None), default=MIN_STYLE_FILE_SIZE_BYTES)

    p.add_argument("--style-cache-size", type=_restricted_int(0, None), default=STYLE_CACHE_SIZE)
    p.add_argument("--nn-algorithm", default=NN_ALGORITHM)
    p.add_argument("--nn-metric", default=NN_METRIC)

    # Camera
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--camera-width", type=_restricted_int(1, None), default=1080)
    p.add_argument("--camera-height", type=_restricted_int(1, None), default=1920)
    p.add_argument("--camera-warmup-frames", type=_restricted_int(0, None), default=15)
    p.add_argument("--camera-grab-frames", type=_restricted_int(0, None), default=5)
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)

    # Display
    p.add_argument("--display-width", type=_restricted_int(1, None), default=1080)
    p.add_argument("--display-height", type=_restricted_int(1, None), default=1920)
    p.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True)

    # Output
    p.add_argument("--save-every-n", type=_restricted_int(0, None), default=0, help="0 disables saving frames.")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--output-prefix", default="frame_nn")

    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=VERBOSE)
    p.add_argument("--tf-xla", action=argparse.BooleanOptionalAction, default=True, help="Enable TF XLA JIT.")
    p.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print timing stats (rolling averages)."
    )
    p.add_argument("--profile-every-n", type=_restricted_int(1, None), default=30)
    return p.parse_args(argv)


def main():
    global IMG_WIDTH, IMG_HEIGHT, STYLE_BATCH_SIZE, VERBOSE, USE_CANNY_PREPROCESS

    args = parse_args()
    VERBOSE = bool(args.verbose)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    IMG_WIDTH = int(args.img_width)
    IMG_HEIGHT = int(args.img_height)
    STYLE_BATCH_SIZE = int(args.style_batch_size)
    USE_CANNY_PREPROCESS = bool(args.use_canny_preprocess)

    if args.tf_xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    encoder = None
    encoder_forward_tf = None
    encoder_forward_np = None
    encoder_forward_batch = None

    if args.encoder_backend == "keras":
        encoder = load_encoder(args.model_path)
        encoder.trainable = False

        @tf.function(reduce_retracing=True)
        def encoder_forward(x: tf.Tensor) -> tf.Tensor:
            return encoder(x, training=False)

        encoder_forward_tf = encoder_forward

        def _forward_batch(x_nhwc_float32: np.ndarray) -> np.ndarray:
            return encoder_forward_tf(tf.convert_to_tensor(x_nhwc_float32)).numpy()

        encoder_forward_batch = _forward_batch
    else:
        encoder_forward_np = load_onnx_encoder(args.onnx_path)
        encoder_forward_batch = encoder_forward_np

    style_index = StyleIndex(
        encoder_forward=encoder_forward_batch,
        style_folder=_resolve_path(args.style_folder),
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        batch_size=STYLE_BATCH_SIZE,
        use_global_avg_pool=args.use_global_avg_pool_for_codes,
        filter_small_files=args.filter_small_style_files,
        min_file_size_bytes=args.min_style_file_size_bytes
    )
    style_codes, style_files = style_index.build()
    style_codes = np.ascontiguousarray(style_codes.astype(np.float32, copy=False))

    nbr = NearestNeighbors(n_neighbors=1, algorithm=args.nn_algorithm, metric=args.nn_metric, n_jobs=-1)
    nbr.fit(style_codes)

    def _load_style_rgb(p: str) -> np.ndarray:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read style image: {p}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    load_style_rgb = lru_cache(maxsize=int(args.style_cache_size))(_load_style_rgb)

    @lru_cache(maxsize=int(args.style_cache_size))
    def get_display_bgr(style_path: str) -> np.ndarray:
        rgb = load_style_rgb(style_path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(
            bgr,
            (int(args.display_width), int(args.display_height)),
            interpolation=cv2.INTER_AREA
        )

    out_dir = _resolve_path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    win = " "
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

    cap = _open_camera(args.camera_index, args.camera_width, args.camera_height)
    try:
        for _ in range(int(args.camera_warmup_frames)):
            cap.read()

        # Warm up once.
        dummy = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        if encoder_forward_tf is not None:
            _ = encoder_forward_tf(tf.convert_to_tensor(dummy))
        else:
            _ = encoder_forward_np(dummy)

        frame_idx = 0
        # rolling timing (ms)
        t_pre_ms = []
        t_inf_ms = []
        t_nn_ms = []
        t_disp_ms = []
        t_total_ms = []

        while True:
            t0 = time.perf_counter()
            # Flush some frames
            for _ in range(int(args.camera_grab_frames)):
                cap.grab()

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            t_cam = time.perf_counter()

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t1 = time.perf_counter()
            enc_in = preprocess_for_encoder(frame_rgb)[None, ...].astype(np.float32, copy=False)
            t2 = time.perf_counter()

            if encoder_forward_tf is not None:
                raw = encoder_forward_tf(tf.convert_to_tensor(enc_in)).numpy()
            else:
                raw = encoder_forward_np(enc_in)
            t3 = time.perf_counter()

            vec = encode_to_feature_vectors(raw, use_global_avg_pool=args.use_global_avg_pool_for_codes)
            t4 = time.perf_counter()
            _, idx = nbr.kneighbors(vec)
            t5 = time.perf_counter()
            style_path = style_files[int(idx[0, 0])]

            disp_bgr = get_display_bgr(style_path)
            cv2.imshow(win, disp_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            t6 = time.perf_counter()

            frame_idx += 1
            if args.profile:
                # per-frame timings
                cam_ms = (t_cam - t0) * 1000.0
                pre_ms = (t2 - t1) * 1000.0
                inf_ms = (t3 - t2) * 1000.0
                nn_ms = (t5 - t4) * 1000.0
                disp_ms = (t6 - t5) * 1000.0
                total_ms = (t6 - t0) * 1000.0

                # camera acquisition time isn't included in other stages
                # but dominates end-to-end latency if the camera delivers low FPS.
                # (e.g., high-res frames from some webcams.)

                # store
                # NOTE: keep arrays aligned for averaging
                # (cam time uses same history window as others)
                # create lazily to avoid changing earlier code structure
                if "t_cam_ms" not in locals():
                    t_cam_ms = []

                t_cam_ms.append(cam_ms)
                t_pre_ms.append(pre_ms)
                t_inf_ms.append(inf_ms)
                t_nn_ms.append(nn_ms)
                t_disp_ms.append(disp_ms)
                t_total_ms.append(total_ms)

                # cap history
                if len(t_total_ms) > 300:
                    t_cam_ms[:] = t_cam_ms[-300:]
                    t_pre_ms[:] = t_pre_ms[-300:]
                    t_inf_ms[:] = t_inf_ms[-300:]
                    t_nn_ms[:] = t_nn_ms[-300:]
                    t_disp_ms[:] = t_disp_ms[-300:]
                    t_total_ms[:] = t_total_ms[-300:]

                if frame_idx % int(args.profile_every_n) == 0:
                    def _avg(xs): return float(np.mean(xs)) if xs else 0.0
                    def _p95(xs): return float(np.percentile(xs, 95)) if xs else 0.0

                    avg_cam = _avg(t_cam_ms)
                    avg_pre, avg_inf, avg_nn, avg_disp, avg_total = map(_avg, [t_pre_ms, t_inf_ms, t_nn_ms, t_disp_ms, t_total_ms])
                    p95_inf = _p95(t_inf_ms)
                    fps = 1000.0 / max(1e-6, avg_total)

                    print(
                        f"[perf] frames={frame_idx} | "
                        f"cam={avg_cam:.1f}ms | "
                        f"pre={avg_pre:.1f}ms | inf={avg_inf:.1f}ms (p95 {p95_inf:.1f}) | "
                        f"nn={avg_nn:.1f}ms | disp={avg_disp:.1f}ms | total={avg_total:.1f}ms | ~{fps:.1f} FPS",
                        flush=True
                    )

            if int(args.save_every_n) > 0 and (frame_idx % int(args.save_every_n) == 0):
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = os.path.join(out_dir, f"{args.output_prefix}_{ts}.jpg")
                cv2.imwrite(out_path, disp_bgr)
                log(f"Saved {out_path}")
    finally:
        cap.release()
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


if __name__ == "__main__":
    main()

