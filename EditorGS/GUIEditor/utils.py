import random


def sample_train_camera(colmap_cameras, edit_cam_num):
    total_view_num = len(colmap_cameras)
    random.seed(0)  # make sure same views
    view_index = random.sample(
        range(0, total_view_num),
        min(total_view_num, edit_cam_num),
    )
    edit_cameras = [colmap_cameras[idx] for idx in view_index]

    return edit_cameras