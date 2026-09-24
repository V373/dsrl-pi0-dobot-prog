"""Render a Dobot rollout beside its query-aligned reward diagnostics."""

from pathlib import Path

import numpy as np


def _expand_query_values(values, lengths):
    repeated = np.repeat(values, lengths, axis=0)
    return np.concatenate((repeated, values[-1:]), axis=0)


def save_dobot_progress_video(
    *,
    raw_video_path,
    output_path,
    query_step_lengths,
    progress_gated,
    is_ood,
    conformal_p_value,
    bin_progress_values,
    rewards,
    is_success,
    reward_type,
    fps,
):
    """Save one control-rate MP4 while holding each query value over its steps."""
    import cv2
    import imageio.v2 as imageio
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    lengths = np.asarray(query_step_lengths)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer) \
            or lengths.size < 1 or np.any(lengths < 1):
        raise ValueError("query_step_lengths must be a nonempty array of positive integers")
    query_count = lengths.size
    control_steps = int(lengths.sum())
    progress = np.asarray(progress_gated, dtype=np.float64)
    ood = np.asarray(is_ood, dtype=bool)
    p_values = np.asarray(conformal_p_value, dtype=np.float64)
    bins = np.asarray(bin_progress_values, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    if progress.shape != (query_count,) or ood.shape != (query_count,) \
            or reward.shape != (query_count,):
        raise ValueError("progress, OOD, and rewards need one value per query")
    if bins.ndim != 1 or bins.size < 2 or not np.all(np.diff(bins) > 0):
        raise ValueError("bin_progress_values must be strictly increasing")
    if p_values.shape != (query_count, bins.size):
        raise ValueError("conformal_p_value must have shape [queries, progress bins]")
    if not np.isfinite(progress).all() or not np.isfinite(p_values).all() \
            or not np.isfinite(bins).all() or not np.isfinite(reward).all():
        raise ValueError("progress video data must be finite")
    if np.any((progress < 0) | (progress > 1)) \
            or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("progress and conformal p-values must be in [0, 1]")
    if reward_type not in ("dense", "pbrs"):
        raise ValueError("reward_type must be dense or pbrs")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")

    progress_frames = _expand_query_values(progress, lengths)
    ood_frames = _expand_query_values(ood, lengths)
    p_value_frames = _expand_query_values(p_values, lengths)
    reward_frames = _expand_query_values(reward, lengths)
    success_frames = np.zeros(control_steps + 1, dtype=np.float64)
    success_frames[-1] = float(is_success)
    frame_steps = np.arange(control_steps + 1)

    capture = cv2.VideoCapture(str(raw_video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open rollout video: {raw_video_path}")
    raw_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if raw_height < 1 or raw_width < 1:
        capture.release()
        raise ValueError("Rollout video has invalid dimensions")
    panel_height = 224
    scaled_width = max(2, round(raw_width * panel_height / raw_height))

    middle = Figure(figsize=(4.1, 2.24), dpi=300)
    middle_canvas = FigureCanvasAgg(middle)
    middle_grid = middle.add_gridspec(
        3, 2, height_ratios=(1, 6, 6), width_ratios=(1, 0.035),
        left=0.11, right=0.90, bottom=0.16, top=0.95,
        hspace=0.25, wspace=0.13,
    )
    ood_axis = middle.add_subplot(middle_grid[0, 0])
    progress_axis = middle.add_subplot(middle_grid[1, 0], sharex=ood_axis)
    p_value_axis = middle.add_subplot(middle_grid[2, 0], sharex=ood_axis)
    colorbar_axis = middle.add_subplot(middle_grid[2, 1])
    ood_axis.imshow(
        ood_frames[np.newaxis, :], aspect="auto", interpolation="nearest",
        cmap=ListedColormap(("white", "#d62728")), vmin=0, vmax=1,
        extent=(-0.5, control_steps + 0.5, 0, 1),
    )
    ood_axis.set_ylabel("OOD", rotation=0, labelpad=17, fontsize=7)
    ood_axis.set_yticks([])
    ood_axis.tick_params(axis="x", bottom=False, labelbottom=False)

    progress_axis.step(frame_steps, progress_frames, where="post", color="#1f77b4", linewidth=1.5)
    progress_axis.set_ylabel("Progress\ngated", fontsize=7)
    progress_axis.set_ylim(-0.02, 1.02)
    progress_axis.set_yticks((0, 0.5, 1))
    progress_axis.tick_params(axis="x", bottom=False, labelbottom=False)

    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    bin_edges = np.concatenate(((max(0.0, bins[0] - (bin_midpoints[0] - bins[0])),),
                                bin_midpoints,
                                (min(1.0, bins[-1] + (bins[-1] - bin_midpoints[-1])),)))
    heatmap = p_value_axis.pcolormesh(
        np.arange(control_steps + 2) - 0.5, bin_edges, p_value_frames.T,
        cmap="Blues", vmin=0, vmax=1, shading="flat",
    )
    p_value_axis.set_ylabel("Progress", fontsize=7)
    p_value_axis.set_xlabel("Control step", fontsize=7)
    p_value_axis.set_ylim(0, 1)
    colorbar = middle.colorbar(heatmap, cax=colorbar_axis)
    colorbar.set_ticks((0, 0.5, 1))
    colorbar.set_label("Conformal p-value", fontsize=6, labelpad=1)
    colorbar.ax.tick_params(labelsize=6, length=2)

    right = Figure(figsize=(3.6, 2.24), dpi=300)
    right_canvas = FigureCanvasAgg(right)
    middle_width, middle_height = middle_canvas.get_width_height()
    right_width, right_height = right_canvas.get_width_height()
    if middle_height != right_height:
        raise RuntimeError("Progress and reward panels must have the same pixel height")
    progress_box = progress_axis.get_position()
    p_value_box = p_value_axis.get_position()
    plot_left = progress_box.x0 * middle_width / right_width
    plot_width = progress_box.width * middle_width / right_width
    success_axis = right.add_axes(
        (plot_left, progress_box.y0, plot_width, progress_box.height))
    reward_axis = right.add_axes(
        (plot_left, p_value_box.y0, plot_width, p_value_box.height),
        sharex=success_axis)
    success_axis.step(frame_steps, success_frames, where="post", color="#9467bd", linewidth=1.5)
    success_axis.plot(
        control_steps, success_frames[-1], "o",
        color="#2ca02c" if is_success else "#d62728", markersize=5,
    )
    success_axis.set_ylabel("Success\nsignal", fontsize=7)
    success_axis.set_ylim(-0.08, 1.08)
    success_axis.set_yticks((0, 1))
    success_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    terminal_label = success_axis.text(
        0.98, 0.85, "",
        ha="right", va="top", transform=success_axis.transAxes,
        color="#2ca02c" if is_success else "#d62728", fontsize=7,
    )
    terminal_label.set_animated(True)
    reward_axis.step(frame_steps, reward_frames, where="post", color="#ff7f0e", linewidth=1.5)
    reward_axis.axhline(0, color="#888888", linewidth=0.7)
    low = min(0.0, float(reward_frames.min()))
    high = max(0.0, float(reward_frames.max()))
    padding = max(0.05, 0.1 * (high - low))
    reward_axis.set_ylim(low - padding, high + padding)
    reward_axis.set_ylabel("PBRS reward" if reward_type == "pbrs" else "Dense reward", fontsize=7)
    reward_axis.set_xlabel("Control step", fontsize=7)

    cursors = []
    for axis in (ood_axis, progress_axis, p_value_axis, success_axis, reward_axis):
        axis.set_xlim(-0.5, control_steps + 0.5)
        axis.tick_params(labelsize=6, length=2)
        cursor = axis.axvline(0, color="black", linestyle="--", linewidth=0.7)
        cursor.set_animated(True)
        cursors.append(cursor)
    middle_canvas.draw()
    right_canvas.draw()
    middle_background = middle_canvas.copy_from_bbox(middle.bbox)
    right_background = right_canvas.copy_from_bbox(right.bbox)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    writer = None
    try:
        writer = imageio.get_writer(
            str(temporary_path), fps=fps, format="ffmpeg", codec="libx264",
            macro_block_size=1,
        )
        for frame_index in frame_steps:
            ok, raw_bgr = capture.read()
            if not ok:
                raise ValueError(f"Rollout video ended before frame {frame_index}")
            terminal_label.set_text(
                ("SUCCESS" if is_success else "FAILURE")
                if frame_index == control_steps else "")
            for cursor in cursors:
                cursor.set_xdata((frame_index, frame_index))
            for canvas, figure, background, panel_cursors in (
                (middle_canvas, middle, middle_background, cursors[:3]),
                (right_canvas, right, right_background, cursors[3:] + [terminal_label]),
            ):
                canvas.restore_region(background)
                for cursor in panel_cursors:
                    cursor.axes.draw_artist(cursor)
                canvas.blit(figure.bbox)
            middle_rgb = cv2.resize(
                np.asarray(middle_canvas.buffer_rgba())[:, :, :3],
                (round(middle_width * panel_height / middle_height), panel_height),
                interpolation=cv2.INTER_AREA,
            )
            right_rgb = cv2.resize(
                np.asarray(right_canvas.buffer_rgba())[:, :, :3],
                (round(right_width * panel_height / right_height), panel_height),
                interpolation=cv2.INTER_AREA,
            )
            raw_rgb = cv2.cvtColor(
                cv2.resize(raw_bgr, (scaled_width, panel_height), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            combined = np.concatenate((raw_rgb, middle_rgb, right_rgb), axis=1)
            if combined.shape[1] % 2:
                combined = combined[:, :-1]
            writer.append_data(np.ascontiguousarray(combined))
        if capture.read()[0]:
            raise ValueError("Rollout video has more frames than the control-step trace")
        writer.close()
        writer = None
        temporary_path.replace(output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        middle.clear()
        right.clear()
    return output_path
