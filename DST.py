import os
import math
import random
import queue
import threading
import argparse
from glob import glob
from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from sklearn.neighbors import NearestNeighbors

# ============================================================
# SETTINGS
# ============================================================
IMG_WIDTH = 256
IMG_HEIGHT = 256

STYLE_BATCH_SIZE = 64
STYLE_FOLDER = "./inputs/Gemini_Generated_Image_am1379am1379am13/quadtree"   # @param {type:"string"}
INPUT_PATH = "./inputs/Gemini_Generated_Image_f9sfhjf9sfhjf9sf-edit.png"          # @param {type:"string"}
MODEL_PATH = "./checkpoints/saved_model.keras"
OUTPUT_NAME = "./outputs/substitution.jpg"

THRESHOLD = 2   # @param {type:"slider"}
MIN_CELL = 6    # @param {type:"slider"}

USE_RANDOM_SPLIT = False
W_RANDOMNESS = 0.5   # @param {type:"slider", min:0, max:0.5, step:0.1}
H_RANDOMNESS = 0.1   # @param {type:"slider", min:0, max:0.5, step:0.1}

USE_MEMMAP_FOR_STYLE_CODES = False
STYLE_CODES_MEMMAP_PATH = "/tmp/style_codes.dat"

STYLE_CACHE_SIZE = 128
PATCH_CACHE_SIZE = 1024
NN_ALGORITHM = "auto"
NN_METRIC = "euclidean"

USE_GLOBAL_AVG_POOL_FOR_CODES = True  # @param {type:"boolean"}

# ------------------------------------------------------------
# STYLE FILE SIZE FILTER
# ------------------------------------------------------------
FILTER_SMALL_STYLE_FILES = True  # @param {type:"boolean"}
MIN_STYLE_FILE_SIZE_BYTES = 1200    # @param {type:"integer"}

# ------------------------------------------------------------
# VIDEO EXPORT
# Direct MP4 writing, no intermediate frame images
# ------------------------------------------------------------
SAVE_SUBSTITUTION_VIDEO = True   # @param {type:"boolean"}
VIDEO_NAME = "./outputs/substitution_animation.mp4"
VIDEO_FPS = 24
VIDEO_CODEC_FOURCC = "mp4v"
SAVE_EVERY_N_SUBSTITUTIONS = 4   # @param {type:"slider"}
VIDEO_QUEUE_MAXSIZE = 128
WRITE_FINAL_FRAME_AT_END = True

# ------------------------------------------------------------
# AGENT SYSTEM
# ------------------------------------------------------------
AGENT_POPULATION = 24            # @param {type:"integer"}
AGENT_NEIGHBOR_K = 24            # @param {type:"integer"}
AGENT_SHUFFLE_EACH_ROUND = False  # @param {type:"boolean"}

SEED = 42

# ============================================================
# DEBUG / LOGGING
# ============================================================
VERBOSE = False  # @param {type:"boolean"}
STYLE_LOG_EVERY = 20
SUBDIVIDE_LOG_EVERY = 5000
AGENT_LOG_EVERY = 250
RENDER_BATCH_DETAIL = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)


# ============================================================
# MODEL LOADING
# ============================================================
def rounded_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def load_encoder(model_path: str):
    print("Loading model...")
    loaded_model = load_model(
        model_path,
        custom_objects={"rounded_accuracy": rounded_accuracy}
    )
    loaded_model.summary()

    encoder_output_layer = loaded_model.layers[-2]
    return tf.keras.models.Model(
        inputs=loaded_model.input,
        outputs=encoder_output_layer.output
    )


# ============================================================
# UTILS
# ============================================================
def ensure_output_dir(input_path: str) -> str:
    filedir, _ = os.path.splitext(input_path)
    os.makedirs(filedir, exist_ok=True)
    return filedir


def preprocess_tile_for_encoder(tile_rgb: np.ndarray) -> np.ndarray:
    tile_resized = cv2.resize(tile_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    tile_float = tile_resized.astype(np.float32) / 255.0
    return tile_float


def encode_to_feature_vectors(codes: np.ndarray, use_global_avg_pool: bool = False) -> np.ndarray:
    """
    Convert encoder output into 2D feature vectors for nearest-neighbor search.
    """
    if codes.ndim == 2:
        return codes.astype(np.float32, copy=False)

    if use_global_avg_pool:
        axes = tuple(range(1, codes.ndim - 1))
        pooled = np.mean(codes, axis=axes, dtype=np.float32)
        return pooled.astype(np.float32, copy=False)

    flat = codes.reshape(codes.shape[0], -1)
    return flat.astype(np.float32, copy=False)


def ensure_even_dimensions(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    if new_h == h and new_w == w:
        return img
    return img[:new_h, :new_w]


# ============================================================
# ASYNC VIDEO WRITER
# ============================================================
class AsyncVideoWriter:
    def __init__(self, output_path: str, frame_size_hw: tuple[int, int], fps: int = 24,
                 fourcc: str = "mp4v", queue_maxsize: int = 32):
        self.output_path = output_path
        self.height, self.width = frame_size_hw
        self.fps = fps
        self.fourcc = fourcc
        self.queue_maxsize = queue_maxsize

        self._queue = queue.Queue(maxsize=queue_maxsize)
        self._writer = None
        self._thread = None
        self._stop_token = object()
        self._error = None
        self.frames_written = 0
        self.frames_dropped = 0

    def start(self):
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            self.output_path,
            fourcc_code,
            self.fps,
            (self.width, self.height)
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for: {self.output_path}")

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        log(
            f"[Video] started | path={self.output_path} | "
            f"size=({self.width}x{self.height}) | fps={self.fps} | fourcc={self.fourcc}"
        )

    def _worker(self):
        try:
            while True:
                item = self._queue.get()
                if item is self._stop_token:
                    self._queue.task_done()
                    break

                frame_rgb = item
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self._writer.write(frame_bgr)
                self.frames_written += 1
                self._queue.task_done()

        except Exception as exc:
            self._error = exc
        finally:
            if self._writer is not None:
                self._writer.release()

    def write(self, frame_rgb: np.ndarray):
        if self._error is not None:
            raise RuntimeError(f"Video writer worker failed: {self._error}")

        frame_rgb = ensure_even_dimensions(frame_rgb)
        if frame_rgb.shape[0] != self.height or frame_rgb.shape[1] != self.width:
            raise ValueError(
                f"Frame size mismatch. Expected {(self.height, self.width)}, got {frame_rgb.shape[:2]}"
            )

        frame_copy = np.ascontiguousarray(frame_rgb.copy())

        try:
            self._queue.put(frame_copy, block=False)
        except queue.Full:
            self.frames_dropped += 1

    def close(self):
        if self._thread is None:
            return

        self._queue.put(self._stop_token)
        self._thread.join()

        if self._error is not None:
            raise RuntimeError(f"Video writer worker failed: {self._error}")

        log(
            f"[Video] complete | written={self.frames_written} | "
            f"dropped={self.frames_dropped} | path={self.output_path}"
        )


# ============================================================
# STYLE INDEX BUILDER
# ============================================================
class StyleIndex:
    def __init__(
        self,
        encoder_model,
        style_folder: str,
        img_width: int = 256,
        img_height: int = 256,
        batch_size: int = 32,
        use_memmap: bool = False,
        memmap_path: str = "/tmp/style_codes.dat",
        use_global_avg_pool: bool = False,
        filter_small_files: bool = False,
        min_file_size_bytes: int = 0
    ):
        self.encoder = encoder_model
        self.style_folder = style_folder
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.use_memmap = use_memmap
        self.memmap_path = memmap_path
        self.use_global_avg_pool = use_global_avg_pool
        self.filter_small_files = filter_small_files
        self.min_file_size_bytes = min_file_size_bytes

        self.style_files = []
        self.style_codes = None

    def _gather_files(self):
        files = sorted(glob(os.path.join(self.style_folder, "*")))
        if not files:
            raise FileNotFoundError(f"No style files found in: {self.style_folder}")

        if not self.filter_small_files:
            log(f"[StyleIndex] file-size filter disabled | candidate_files={len(files)}")
            return files

        filtered_files = []
        removed_count = 0
        removed_bytes_total = 0

        for f in files:
            try:
                file_size = os.path.getsize(f)
            except OSError:
                removed_count += 1
                continue

            if file_size >= self.min_file_size_bytes:
                filtered_files.append(f)
            else:
                removed_count += 1
                removed_bytes_total += file_size

        if not filtered_files:
            raise RuntimeError(
                f"All style files were filtered out by min_file_size_bytes={self.min_file_size_bytes}"
            )

        log(
            f"[StyleIndex] file-size filter enabled | "
            f"kept={len(filtered_files)}/{len(files)} | "
            f"removed={removed_count} | min_bytes={self.min_file_size_bytes} | "
            f"removed_bytes_total={removed_bytes_total}"
        )

        return filtered_files

    def _load_and_preprocess(self, path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width], antialias=True)
        img.set_shape((self.img_height, self.img_width, 3))
        return img, path

    def _build_dataset(self, files):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self._load_and_preprocess, num_parallel_calls=2)
        ds = ds.ignore_errors()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    def build(self):
        print("Building style index...")
        candidate_files = self._gather_files()
        ds = self._build_dataset(candidate_files)

        valid_paths = []
        initialized = False
        processed_count = 0
        batch_idx = 0

        all_codes = None
        style_codes_memmap = None

        for batch_imgs, batch_paths in ds:
            batch_idx += 1

            raw_codes = self.encoder.predict(batch_imgs, verbose=0)
            batch_codes = encode_to_feature_vectors(
                raw_codes,
                use_global_avg_pool=self.use_global_avg_pool
            )
            batch_paths_decoded = [p.decode("utf-8") for p in batch_paths.numpy()]

            if not initialized:
                latent_dim = batch_codes.shape[1]

                if self.use_memmap:
                    max_items = len(candidate_files)
                    style_codes_memmap = np.memmap(
                        self.memmap_path,
                        dtype=np.float32,
                        mode="w+",
                        shape=(max_items, latent_dim)
                    )
                else:
                    all_codes = []

                initialized = True
                log(
                    f"[StyleIndex] first batch | imgs={tuple(batch_imgs.shape)} | "
                    f"codes={tuple(batch_codes.shape)} | latent_dim={latent_dim}"
                )

            if self.use_memmap:
                next_count = processed_count + len(batch_paths_decoded)
                style_codes_memmap[processed_count:next_count] = batch_codes
                processed_count = next_count
            else:
                all_codes.append(batch_codes)
                processed_count += len(batch_paths_decoded)

            valid_paths.extend(batch_paths_decoded)

            if batch_idx % STYLE_LOG_EVERY == 0:
                log(
                    f"[StyleIndex] batch={batch_idx} | "
                    f"processed={processed_count}/{len(candidate_files)} | "
                    f"last_batch={len(batch_paths_decoded)}"
                )

        if not initialized or len(valid_paths) == 0:
            raise RuntimeError("No decodable style images found.")

        if self.use_memmap:
            self.style_codes = style_codes_memmap[:processed_count]
        else:
            self.style_codes = np.concatenate(all_codes, axis=0)

        self.style_files = valid_paths

        log(
            f"[StyleIndex] complete | indexed={len(self.style_files)} | "
            f"STYLE_CODES shape={self.style_codes.shape} | dtype={self.style_codes.dtype}"
        )

        return self.style_codes, self.style_files


# ============================================================
# QUADTREE
# ============================================================
class Node:
    __slots__ = ("x0", "y0", "width", "height", "children")

    def __init__(self, x0: int, y0: int, width: int, height: int):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.children = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


class QTree:
    def __init__(
        self,
        img: np.ndarray,
        threshold: float,
        min_pixel_size: int,
        use_random_split: bool = False,
        w_randomness: float = 0.0,
        h_randomness: float = 0.0
    ):
        self.img = img
        self.threshold = threshold
        self.min_size = min_pixel_size
        self.use_random_split = use_random_split
        self.w_randomness = w_randomness
        self.h_randomness = h_randomness
        self.root = Node(0, 0, img.shape[1], img.shape[0])

    def get_region(self, node: Node) -> np.ndarray:
        return self.img[node.y0:node.y0 + node.height, node.x0:node.x0 + node.width]

    def get_error(self, node: Node) -> float:
        pixels = self.get_region(node)
        if pixels.size == 0:
            return 0.0

        pixels_f = pixels.astype(np.float32, copy=False)

        b = pixels_f[:, :, 0]
        g = pixels_f[:, :, 1]
        r = pixels_f[:, :, 2]

        b_mean = b.mean()
        g_mean = g.mean()
        r_mean = r.mean()

        b_var = ((b - b_mean) ** 2).mean()
        g_var = ((g - g_mean) ** 2).mean()
        r_var = ((r - r_mean) ** 2).mean()

        weighted_error = r_var * 0.2989 + g_var * 0.5870 + b_var * 0.1140
        scale_factor = (self.img.shape[0] * self.img.shape[1]) / 90000000.0
        return float(weighted_error * scale_factor)

    def split_node(self, node: Node) -> bool:
        if node.width < 2 * self.min_size or node.height < 2 * self.min_size:
            return False

        if self.use_random_split:
            w1 = max(
                math.floor(node.width * random.uniform(0.5 - self.w_randomness, 0.5 + self.w_randomness)),
                self.min_size
            )
            h1 = max(
                math.floor(node.height * random.uniform(0.5 - self.h_randomness, 0.5 + self.h_randomness)),
                self.min_size
            )
        else:
            w1 = max(node.width // 2, self.min_size)
            h1 = max(node.height // 2, self.min_size)

        w2 = node.width - w1
        h2 = node.height - h1

        if w2 < self.min_size or h2 < self.min_size:
            return False

        node.children = [
            Node(node.x0, node.y0, w1, h1),
            Node(node.x0, node.y0 + h1, w1, h2),
            Node(node.x0 + w1, node.y0, w2, h1),
            Node(node.x0 + w1, node.y0 + h1, w2, h2),
        ]
        return True

    def subdivide(self):
        print("Subdividing quadtree...")
        stack = [self.root]
        processed_nodes = 0
        split_nodes = 0
        max_stack_size = 1

        while stack:
            node = stack.pop()
            processed_nodes += 1

            if node.width < 2 * self.min_size or node.height < 2 * self.min_size:
                if processed_nodes % SUBDIVIDE_LOG_EVERY == 0:
                    log(
                        f"[QTree] processed={processed_nodes} | splits={split_nodes} | "
                        f"stack={len(stack)} | max_stack={max_stack_size}"
                    )
                continue

            if self.get_error(node) <= self.threshold:
                if processed_nodes % SUBDIVIDE_LOG_EVERY == 0:
                    log(
                        f"[QTree] processed={processed_nodes} | splits={split_nodes} | "
                        f"stack={len(stack)} | max_stack={max_stack_size}"
                    )
                continue

            if self.split_node(node):
                stack.extend(node.children)
                split_nodes += 1
                if len(stack) > max_stack_size:
                    max_stack_size = len(stack)

            if processed_nodes % SUBDIVIDE_LOG_EVERY == 0:
                log(
                    f"[QTree] processed={processed_nodes} | splits={split_nodes} | "
                    f"stack={len(stack)} | max_stack={max_stack_size}"
                )

        log(f"[QTree] complete | processed={processed_nodes} | splits={split_nodes}")

    def iter_leaves(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                yield node
            else:
                stack.extend(node.children)


# ============================================================
# TILE / AGENT STRUCTURES
# ============================================================
class TileRecord:
    __slots__ = (
        "tile_id", "node", "x0", "y0", "width", "height",
        "cx", "cy", "area", "claimed_by", "substituted", "neighbor_ids"
    )

    def __init__(self, tile_id: int, node: Node):
        self.tile_id = tile_id
        self.node = node
        self.x0 = node.x0
        self.y0 = node.y0
        self.width = node.width
        self.height = node.height
        self.cx = node.x0 + node.width / 2.0
        self.cy = node.y0 + node.height / 2.0
        self.area = node.width * node.height
        self.claimed_by = -1
        self.substituted = False
        self.neighbor_ids = []


class Agent:
    __slots__ = ("agent_id", "current_tile_id", "claimed_tiles", "active")

    def __init__(self, agent_id: int, start_tile_id: int):
        self.agent_id = agent_id
        self.current_tile_id = start_tile_id
        self.claimed_tiles = 0
        self.active = True


# ============================================================
# AGENT-BASED TERRITORIAL SUBSTITUTION ENGINE
# ============================================================
class TerritorialSubstitutionEngine:
    def __init__(
        self,
        img: np.ndarray,
        encoder_model,
        style_codes: np.ndarray,
        style_files,
        use_global_avg_pool: bool = False,
        style_cache_size: int = 128,
        patch_cache_size: int = 1024,
        nn_algorithm: str = "auto",
        nn_metric: str = "euclidean",
        save_video: bool = False,
        video_writer: AsyncVideoWriter | None = None,
        save_every_n_substitutions: int = 10,
        write_final_frame_at_end: bool = True,
        agent_population: int = 24,
        agent_neighbor_k: int = 24,
        agent_shuffle_each_round: bool = True
    ):
        self.img = img
        self.encoder = encoder_model
        self.style_codes = style_codes
        self.style_files = list(style_files)
        self.use_global_avg_pool = use_global_avg_pool

        self.save_video = save_video
        self.video_writer = video_writer
        self.save_every_n_substitutions = max(1, int(save_every_n_substitutions))
        self.write_final_frame_at_end = write_final_frame_at_end

        self.agent_population = max(1, int(agent_population))
        self.agent_neighbor_k = max(2, int(agent_neighbor_k))
        self.agent_shuffle_each_round = agent_shuffle_each_round

        self.substitution_counter = 0

        self.nbr = NearestNeighbors(
            n_neighbors=1,
            algorithm=nn_algorithm,
            metric=nn_metric
        )
        self.nbr.fit(self.style_codes)

        self._load_style_rgb = lru_cache(maxsize=style_cache_size)(self._load_style_rgb_uncached)
        self._get_resized_patch = lru_cache(maxsize=patch_cache_size)(self._get_resized_patch_uncached)

    @staticmethod
    def _load_style_rgb_uncached(style_path: str) -> np.ndarray:
        img_bgr = cv2.imread(style_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not read style image: {style_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _get_resized_patch_uncached(self, style_path: str, width: int, height: int) -> np.ndarray:
        style_img = self._load_style_rgb(style_path)
        return cv2.resize(style_img, (width, height), interpolation=cv2.INTER_AREA)

    def _maybe_write_video_frame(self, current_rgb: np.ndarray):
        if not self.save_video or self.video_writer is None:
            return

        if self.substitution_counter % self.save_every_n_substitutions == 0:
            frame_rgb = ensure_even_dimensions(current_rgb)
            self.video_writer.write(frame_rgb)

    def _build_tiles(self, leaves):
        tiles = [TileRecord(i, node) for i, node in enumerate(leaves)]
        return tiles

    def _build_neighbor_graph(self, tiles):
        coords = np.array([[t.cx, t.cy] for t in tiles], dtype=np.float32)
        n_tiles = len(tiles)
        k = min(self.agent_neighbor_k + 1, n_tiles)

        if n_tiles <= 1:
            return

        centroid_nbr = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        centroid_nbr.fit(coords)
        _, indices = centroid_nbr.kneighbors(coords)

        for i, tile in enumerate(tiles):
            tile.neighbor_ids = [int(j) for j in indices[i] if int(j) != i]

    def _seed_agent_tile_ids(self, tiles):
        areas = np.array([t.area for t in tiles], dtype=np.int64)
        sorted_ids = np.argsort(areas)

        min_area = areas[sorted_ids[0]]
        smallest_ids = [int(i) for i in sorted_ids if areas[i] == min_area]

        if len(smallest_ids) >= self.agent_population:
            return random.sample(smallest_ids, self.agent_population)

        # If not enough strictly-smallest tiles, expand upward through the smallest set
        seed_pool = [int(i) for i in sorted_ids[:min(self.agent_population * 4, len(sorted_ids))]]
        seed_count = min(self.agent_population, len(seed_pool))
        return random.sample(seed_pool, seed_count)

    def _claim_tile(self, tiles, tile_id: int, agent_id: int):
        tile = tiles[tile_id]
        if tile.claimed_by != -1:
            return False
        tile.claimed_by = agent_id
        return True

    def _choose_next_tile_for_agent(self, agent: Agent, tiles, unclaimed_ids: set[int]):
        if not unclaimed_ids:
            return None

        current_tile = tiles[agent.current_tile_id]

        # First preference: nearby precomputed neighbors not yet claimed
        for neighbor_id in current_tile.neighbor_ids:
            if neighbor_id in unclaimed_ids:
                return neighbor_id

        # Fallback: nearest remaining unclaimed tile by centroid distance
        best_id = None
        best_dist2 = None
        cx, cy = current_tile.cx, current_tile.cy

        for tile_id in unclaimed_ids:
            tile = tiles[tile_id]
            dx = tile.cx - cx
            dy = tile.cy - cy
            dist2 = dx * dx + dy * dy
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_id = tile_id

        return best_id

    def _substitute_tile_batch(self, img_out: np.ndarray, tiles, tile_ids):
        if not tile_ids:
            return 0

        batch_tiles = []
        valid_tile_ids = []

        for tile_id in tile_ids:
            tile = tiles[tile_id]
            crop = self.img[tile.y0:tile.y0 + tile.height, tile.x0:tile.x0 + tile.width]
            if crop.size == 0:
                continue
            batch_tiles.append(preprocess_tile_for_encoder(crop))
            valid_tile_ids.append(tile_id)

        if not valid_tile_ids:
            return 0

        batch = np.stack(batch_tiles, axis=0).astype(np.float32)
        raw_codes = self.encoder.predict(batch, verbose=0)
        codes = encode_to_feature_vectors(
            raw_codes,
            use_global_avg_pool=self.use_global_avg_pool
        )

        distances, indices = self.nbr.kneighbors(codes)
        nn_indices = indices[:, 0]

        if RENDER_BATCH_DETAIL:
            mean_dist = float(np.mean(distances))
            log(
                f"[AgentBatch] batch_size={len(valid_tile_ids)} | "
                f"code_shape={tuple(codes.shape)} | mean_nn_dist={mean_dist:.6f}"
            )

        substituted = 0

        for tile_id, nn_idx in zip(valid_tile_ids, nn_indices):
            tile = tiles[tile_id]
            if nn_idx < 0 or nn_idx >= len(self.style_files):
                continue

            style_path = self.style_files[int(nn_idx)]

            try:
                patch = self._get_resized_patch(style_path, tile.width, tile.height)
            except Exception as exc:
                print(f"Warning: failed to prepare style patch '{style_path}': {exc}")
                continue

            img_out[tile.y0:tile.y0 + tile.height, tile.x0:tile.x0 + tile.width] = patch
            tile.substituted = True
            substituted += 1
            self.substitution_counter += 1
            self._maybe_write_video_frame(img_out)

        return substituted

    def run(self, leaves) -> np.ndarray:
        print("Running agent-based territorial substitution...")
        img_out = self.img.copy()

        tiles = self._build_tiles(leaves)
        self._build_neighbor_graph(tiles)

        if not tiles:
            return img_out

        seed_tile_ids = self._seed_agent_tile_ids(tiles)
        agents = []

        unclaimed_ids = set(range(len(tiles)))

        for agent_id, seed_tile_id in enumerate(seed_tile_ids):
            if seed_tile_id not in unclaimed_ids:
                continue
            if not self._claim_tile(tiles, seed_tile_id, agent_id):
                continue
            unclaimed_ids.remove(seed_tile_id)
            agents.append(Agent(agent_id=agent_id, start_tile_id=seed_tile_id))

        if not agents:
            raise RuntimeError("No agents were initialized. Check subdivision and agent population settings.")

        print(f"Tile count: {len(tiles)}")
        print(f"Agent count: {len(agents)}")

        # Initial substitution at agent seed positions
        substituted_total = self._substitute_tile_batch(
            img_out=img_out,
            tiles=tiles,
            tile_ids=[a.current_tile_id for a in agents]
        )
        for agent in agents:
            agent.claimed_tiles += 1

        rounds = 0

        while unclaimed_ids:
            rounds += 1
            progress_made = False

            active_agents = [a for a in agents if a.active]
            if not active_agents:
                break

            if self.agent_shuffle_each_round:
                random.shuffle(active_agents)

            planned_moves = []
            planned_tile_ids = []

            for agent in active_agents:
                next_tile_id = self._choose_next_tile_for_agent(agent, tiles, unclaimed_ids)

                if next_tile_id is None:
                    agent.active = False
                    continue

                if next_tile_id not in unclaimed_ids:
                    continue

                # Reserve tile immediately so no two agents choose the same tile
                claimed = self._claim_tile(tiles, next_tile_id, agent.agent_id)
                if not claimed:
                    continue

                unclaimed_ids.remove(next_tile_id)
                planned_moves.append((agent, next_tile_id))
                planned_tile_ids.append(next_tile_id)

            if planned_tile_ids:
                substituted_now = self._substitute_tile_batch(
                    img_out=img_out,
                    tiles=tiles,
                    tile_ids=planned_tile_ids
                )
                substituted_total += substituted_now

                for agent, next_tile_id in planned_moves:
                    agent.current_tile_id = next_tile_id
                    agent.claimed_tiles += 1
                    progress_made = True

            if rounds % AGENT_LOG_EVERY == 0:
                extra = ""
                if self.save_video and self.video_writer is not None:
                    extra = (
                        f" | video_written={self.video_writer.frames_written}"
                        f" | video_dropped={self.video_writer.frames_dropped}"
                    )

                log(
                    f"[Agents] round={rounds} | "
                    f"claimed={len(tiles) - len(unclaimed_ids)}/{len(tiles)} | "
                    f"remaining={len(unclaimed_ids)} | "
                    f"substituted={substituted_total}"
                    f"{extra}"
                )

            if not progress_made:
                # Safety fallback: if agents stall but tiles remain, assign nearest unclaimed tiles
                stalled_agents = [a for a in agents if a.active]
                if not stalled_agents:
                    break

                fallback_moves = []
                fallback_tile_ids = []

                for agent in stalled_agents:
                    if not unclaimed_ids:
                        break
                    next_tile_id = self._choose_next_tile_for_agent(agent, tiles, unclaimed_ids)
                    if next_tile_id is None:
                        continue
                    if not self._claim_tile(tiles, next_tile_id, agent.agent_id):
                        continue
                    unclaimed_ids.remove(next_tile_id)
                    fallback_moves.append((agent, next_tile_id))
                    fallback_tile_ids.append(next_tile_id)

                if fallback_tile_ids:
                    substituted_now = self._substitute_tile_batch(
                        img_out=img_out,
                        tiles=tiles,
                        tile_ids=fallback_tile_ids
                    )
                    substituted_total += substituted_now
                    for agent, next_tile_id in fallback_moves:
                        agent.current_tile_id = next_tile_id
                        agent.claimed_tiles += 1
                else:
                    break

        if self.save_video and self.video_writer is not None and self.write_final_frame_at_end:
            frame_rgb = ensure_even_dimensions(img_out)
            self.video_writer.write(frame_rgb)

        extra = ""
        if self.save_video and self.video_writer is not None:
            extra = (
                f" | video_written={self.video_writer.frames_written}"
                f" | video_dropped={self.video_writer.frames_dropped}"
            )

        log(
            f"[Agents] complete | rounds={rounds} | "
            f"claimed={len(tiles) - len(unclaimed_ids)}/{len(tiles)} | "
            f"substituted={substituted_total}"
            f"{extra}"
        )

        return img_out


# ============================================================
# MAIN
# ============================================================
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent-based territorial style substitution (DST).")

    # Image / model sizing & batching
    p.add_argument("--img-width", type=_restricted_int(1, None), default=IMG_WIDTH)
    p.add_argument("--img-height", type=_restricted_int(1, None), default=IMG_HEIGHT)
    p.add_argument("--style-batch-size", type=_restricted_int(1, None), default=STYLE_BATCH_SIZE)

    # Paths / I/O
    p.add_argument("--style-folder", default=STYLE_FOLDER, help="Folder containing style images.")
    p.add_argument("--input-path", default=INPUT_PATH, help="Input image path.")
    p.add_argument("--model-path", default=MODEL_PATH, help="Keras model path.")
    p.add_argument("--output-name", default=OUTPUT_NAME, help="Output filename (joined under input-derived folder).")

    # Quadtree sliders
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Subdivision threshold.")
    p.add_argument("--min-cell", type=_restricted_int(1, None), default=MIN_CELL, help="Minimum cell size (pixels).")
    p.add_argument("--use-random-split", action="store_true", default=USE_RANDOM_SPLIT, help="Enable randomized splits.")
    p.add_argument(
        "--w-randomness",
        type=_restricted_float(0.0, 0.5),
        default=W_RANDOMNESS,
        help="Randomness factor for width split (0..0.5)."
    )
    p.add_argument(
        "--h-randomness",
        type=_restricted_float(0.0, 0.5),
        default=H_RANDOMNESS,
        help="Randomness factor for height split (0..0.5)."
    )

    # Style encoding / filtering
    p.add_argument(
        "--use-global-avg-pool-for-codes",
        action=argparse.BooleanOptionalAction,
        default=USE_GLOBAL_AVG_POOL_FOR_CODES,
        help="Use global average pooling on encoder codes before NN search."
    )
    p.add_argument(
        "--use-memmap-for-style-codes",
        action=argparse.BooleanOptionalAction,
        default=USE_MEMMAP_FOR_STYLE_CODES
    )
    p.add_argument("--style-codes-memmap-path", default=STYLE_CODES_MEMMAP_PATH)

    p.add_argument("--style-cache-size", type=_restricted_int(0, None), default=STYLE_CACHE_SIZE)
    p.add_argument("--patch-cache-size", type=_restricted_int(0, None), default=PATCH_CACHE_SIZE)
    p.add_argument("--nn-algorithm", default=NN_ALGORITHM)
    p.add_argument("--nn-metric", default=NN_METRIC)

    p.add_argument(
        "--filter-small-style-files",
        action=argparse.BooleanOptionalAction,
        default=FILTER_SMALL_STYLE_FILES,
        help="Filter out small/invalid style files by size."
    )
    p.add_argument(
        "--min-style-file-size-bytes",
        type=_restricted_int(0, None),
        default=MIN_STYLE_FILE_SIZE_BYTES,
        help="Minimum size (bytes) for style files when filtering is enabled."
    )

    # Video export
    p.add_argument(
        "--save-substitution-video",
        action=argparse.BooleanOptionalAction,
        default=SAVE_SUBSTITUTION_VIDEO,
        help="Write MP4 animation while substituting tiles."
    )
    p.add_argument("--video-name", default=VIDEO_NAME, help="MP4 output name (under input-derived folder).")
    p.add_argument("--video-fps", type=_restricted_int(1, None), default=VIDEO_FPS)
    p.add_argument("--video-codec-fourcc", default=VIDEO_CODEC_FOURCC)
    p.add_argument("--video-queue-maxsize", type=_restricted_int(1, None), default=VIDEO_QUEUE_MAXSIZE)
    p.add_argument(
        "--save-every-n-substitutions",
        type=_restricted_int(1, None),
        default=SAVE_EVERY_N_SUBSTITUTIONS,
        help="Write a video frame every N substitutions."
    )
    p.add_argument(
        "--write-final-frame-at-end",
        action=argparse.BooleanOptionalAction,
        default=WRITE_FINAL_FRAME_AT_END
    )

    # Agent system
    p.add_argument("--agent-population", type=_restricted_int(1, None), default=AGENT_POPULATION)
    p.add_argument("--agent-neighbor-k", type=_restricted_int(2, None), default=AGENT_NEIGHBOR_K)
    p.add_argument(
        "--agent-shuffle-each-round",
        action=argparse.BooleanOptionalAction,
        default=AGENT_SHUFFLE_EACH_ROUND
    )

    # Misc
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=VERBOSE)
    p.add_argument("--style-log-every", type=_restricted_int(1, None), default=STYLE_LOG_EVERY)
    p.add_argument("--subdivide-log-every", type=_restricted_int(1, None), default=SUBDIVIDE_LOG_EVERY)
    p.add_argument("--agent-log-every", type=_restricted_int(1, None), default=AGENT_LOG_EVERY)
    p.add_argument(
        "--render-batch-detail",
        action=argparse.BooleanOptionalAction,
        default=RENDER_BATCH_DETAIL
    )

    return p.parse_args(argv)


def main(args: argparse.Namespace):
    global IMG_WIDTH, IMG_HEIGHT
    global STYLE_BATCH_SIZE
    global USE_MEMMAP_FOR_STYLE_CODES, STYLE_CODES_MEMMAP_PATH
    global STYLE_CACHE_SIZE, PATCH_CACHE_SIZE, NN_ALGORITHM, NN_METRIC
    global USE_GLOBAL_AVG_POOL_FOR_CODES
    global FILTER_SMALL_STYLE_FILES, MIN_STYLE_FILE_SIZE_BYTES
    global SAVE_SUBSTITUTION_VIDEO, VIDEO_NAME, VIDEO_FPS, VIDEO_CODEC_FOURCC, SAVE_EVERY_N_SUBSTITUTIONS
    global VIDEO_QUEUE_MAXSIZE, WRITE_FINAL_FRAME_AT_END
    global AGENT_POPULATION, AGENT_NEIGHBOR_K, AGENT_SHUFFLE_EACH_ROUND
    global SEED
    global VERBOSE, STYLE_LOG_EVERY, SUBDIVIDE_LOG_EVERY, AGENT_LOG_EVERY, RENDER_BATCH_DETAIL

    IMG_WIDTH = int(args.img_width)
    IMG_HEIGHT = int(args.img_height)
    STYLE_BATCH_SIZE = int(args.style_batch_size)

    USE_MEMMAP_FOR_STYLE_CODES = bool(args.use_memmap_for_style_codes)
    STYLE_CODES_MEMMAP_PATH = str(args.style_codes_memmap_path)

    STYLE_CACHE_SIZE = int(args.style_cache_size)
    PATCH_CACHE_SIZE = int(args.patch_cache_size)
    NN_ALGORITHM = str(args.nn_algorithm)
    NN_METRIC = str(args.nn_metric)

    USE_GLOBAL_AVG_POOL_FOR_CODES = bool(args.use_global_avg_pool_for_codes)

    FILTER_SMALL_STYLE_FILES = bool(args.filter_small_style_files)
    MIN_STYLE_FILE_SIZE_BYTES = int(args.min_style_file_size_bytes)

    SAVE_SUBSTITUTION_VIDEO = bool(args.save_substitution_video)
    VIDEO_NAME = str(args.video_name)
    VIDEO_FPS = int(args.video_fps)
    VIDEO_CODEC_FOURCC = str(args.video_codec_fourcc)
    SAVE_EVERY_N_SUBSTITUTIONS = int(args.save_every_n_substitutions)
    VIDEO_QUEUE_MAXSIZE = int(args.video_queue_maxsize)
    WRITE_FINAL_FRAME_AT_END = bool(args.write_final_frame_at_end)

    AGENT_POPULATION = int(args.agent_population)
    AGENT_NEIGHBOR_K = int(args.agent_neighbor_k)
    AGENT_SHUFFLE_EACH_ROUND = bool(args.agent_shuffle_each_round)

    SEED = int(args.seed)
    VERBOSE = bool(args.verbose)
    STYLE_LOG_EVERY = int(args.style_log_every)
    SUBDIVIDE_LOG_EVERY = int(args.subdivide_log_every)
    AGENT_LOG_EVERY = int(args.agent_log_every)
    RENDER_BATCH_DETAIL = bool(args.render_batch_detail)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    encoder = load_encoder(args.model_path)

    print("Loading input image...")
    img_bgr = cv2.imread(args.input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {args.input_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    output_dir = ""

    style_index = StyleIndex(
        encoder_model=encoder,
        style_folder=args.style_folder,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        batch_size=STYLE_BATCH_SIZE,
        use_memmap=USE_MEMMAP_FOR_STYLE_CODES,
        memmap_path=STYLE_CODES_MEMMAP_PATH,
        use_global_avg_pool=args.use_global_avg_pool_for_codes,
        filter_small_files=args.filter_small_style_files,
        min_file_size_bytes=args.min_style_file_size_bytes
    )
    style_codes, style_files = style_index.build()

    if len(style_files) == 0:
        raise RuntimeError("No valid style images available.")

    qt = QTree(
        img=img_rgb,
        threshold=args.threshold,
        min_pixel_size=args.min_cell,
        use_random_split=args.use_random_split,
        w_randomness=args.w_randomness,
        h_randomness=args.h_randomness
    )
    qt.subdivide()

    leaves = list(qt.iter_leaves())
    print(f"Leaf count: {len(leaves)}")

    video_writer = None
    if args.save_substitution_video:
        video_output_path = os.path.join(output_dir, args.video_name)
        even_img = ensure_even_dimensions(img_rgb)
        video_writer = AsyncVideoWriter(
            output_path=video_output_path,
            frame_size_hw=even_img.shape[:2],
            fps=args.video_fps,
            fourcc=args.video_codec_fourcc,
            queue_maxsize=args.video_queue_maxsize
        )
        video_writer.start()

    engine = TerritorialSubstitutionEngine(
        img=img_rgb,
        encoder_model=encoder,
        style_codes=style_codes,
        style_files=style_files,
        use_global_avg_pool=args.use_global_avg_pool_for_codes,
        style_cache_size=STYLE_CACHE_SIZE,
        patch_cache_size=PATCH_CACHE_SIZE,
        nn_algorithm=NN_ALGORITHM,
        nn_metric=NN_METRIC,
        save_video=args.save_substitution_video,
        video_writer=video_writer,
        save_every_n_substitutions=args.save_every_n_substitutions,
        write_final_frame_at_end=args.write_final_frame_at_end,
        agent_population=args.agent_population,
        agent_neighbor_k=args.agent_neighbor_k,
        agent_shuffle_each_round=args.agent_shuffle_each_round
    )

    try:
        result_rgb = engine.run(leaves)
    finally:
        if video_writer is not None:
            video_writer.close()

    output_path = os.path.join(output_dir, args.output_name)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(output_path, result_bgr)
    if not ok:
        raise IOError(f"Failed to save output image to: {output_path}")

    print(f"Saved output image to: {output_path}")

    if args.save_substitution_video:
        print(f"Saved animation to: {os.path.join(output_dir, args.video_name)}")


if __name__ == "__main__":
    main(parse_args())