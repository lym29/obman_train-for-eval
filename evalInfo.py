import argparse
import json
import os

import numpy as np
import trimesh
from joblib import Parallel, delayed

from mano_train.simulation.simulate import process_sample
from mano_train.netscripts.intersect import get_sample_intersect_volume
from mano_train.networks.branches.contactloss import mesh_vert_int_exts

import pickle
from tqdm import tqdm
import open3d as o3d
import torch



def get_closed_faces(th_faces):
    close_faces = np.array(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    closed_faces = np.concatenate([th_faces, close_faces], axis=0)
    # Indices of faces added during closing --> should be ignored as they match the wrist
    # part of the hand, which is not an external surface of the human

    # Valid because added closed faces are at the end
    hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]

    return closed_faces, hand_ignore_faces

def dict_for_connected_faces(faces):
    v2f = {}
    for f_id, face in enumerate(faces):
        for v in face:
            v2f.setdefault(v, []).append(f_id)
    return v2f

def geneSampleInfos_quick(data):
    device = o3d.core.Device('cuda' if torch.cuda.is_available() else 'CPU:0')
    for k in data:
        if k == "file_name":
            continue
        data[k] = torch.Tensor(np.stack(data[k]))
    hand_verts = data["hand_verts"]
    obj_verts = data["object_verts"]
    obj_faces = data["object_faces"]



def geneSampleInfos(fname_lists, hand_verts, hand_faces, object_verts, object_faces, scale=1):
    """
    Args:
        scale (float): convert to meters
    """
    # object_faces = [ obj_face[:, ::-1]for obj_face in object_faces]  # CCW to CW

    sample_infos = []

    for hand_vert, hand_face, obj_vert, obj_face in tqdm(zip(
        hand_verts, hand_faces, object_verts, object_faces
    )):
        sample_info = {
            "file_names": fname_lists,
            "hand_verts": hand_vert * scale,
            "hand_faces": hand_face,
            "obj_verts": obj_vert * scale,
            "obj_faces": obj_face,
        }

        obj_mesh = trimesh.load({"vertices": obj_vert, "faces": obj_face})
        trimesh.repair.fix_normals(obj_mesh)
        result_close, result_distance, _ = trimesh.proximity.closest_point(
            obj_mesh, hand_vert
        )
        penetrating, exterior = mesh_vert_int_exts(
        obj_mesh, hand_vert, result_distance
        )
        sample_info["max_depth"] = 0 if penetrating.sum() == 0 else result_distance[penetrating == 1].max()

        sample_infos.append(sample_info)

        break

    return sample_infos

def para_simulate(
    sample_infos,  
    saved_path,
    wait_time=0,
    sample_vis_freq=100,
    use_gui=False,
    workers=8,
    cluster=False,
    vhacd_exe=None
):
    save_gif_folder = os.path.join(saved_path, "save_gifs")
    save_obj_folder = os.path.join(saved_path, "save_objs")
    simulation_results_folder = os.path.join(saved_path,"simulation_results")
    os.makedirs(save_gif_folder, exist_ok=True)
    os.makedirs(save_obj_folder, exist_ok=True)
    os.makedirs(simulation_results_folder, exist_ok=True)

    max_depths = [sample_info["max_depth"] for sample_info in sample_infos]
    file_names = [sample_info["file_names"] for sample_info in sample_infos]
    distances = Parallel(n_jobs=workers)(
        delayed(process_sample)(
            sample_idx,
            sample_info,
            save_gif_folder=save_gif_folder,
            save_obj_folder=save_obj_folder,
            use_gui=use_gui,
            wait_time=wait_time,
            sample_vis_freq=sample_vis_freq,
            vhacd_exe=vhacd_exe,
        )
        for sample_idx, sample_info in enumerate(sample_infos)
    )

    volumes = Parallel(n_jobs=workers, verbose=5)(
        delayed(get_sample_intersect_volume)(sample_info)
        for sample_info in sample_infos
    )
    
    simulation_results_path = os.path.join(simulation_results_folder,"results.json")
    with open(simulation_results_path, "w") as j_f:
        json.dump(
            {
                "smi_dists": distances,
                "mean_smi_dist": np.mean(distances),
                "std_smi_dist": np.std(distances),

                "max_depths": max_depths,
                "mean_max_depth": np.mean(max_depths),

                "volumes": volumes,
                "mean_volume": np.mean(volumes),

                "file_names": file_names,

            },
            j_f,
        )
        print("Wrote results to {}".format(simulation_results_path))


if __name__ == "__main__":

    # data_path = "/ghome/l5/ymliu/results/oakink/baseline_0414/test_grasp_results/A01002/0/"
    # obj_file = os.path.join(data_path, "obj_mesh_0.ply")
    # hand_file = os.path.join(data_path, "rh_mesh_rec_0.ply")

    work_dir = "/ghome/l5/ymliu/results/oakink/incre_0414_ds/"
    data_path = os.path.join(work_dir, "for_eval", "incre_results_on_test.pkl")
    saved_path = os.path.join(work_dir, "for_eval")

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    hand_faces_list = []
    for th_faces in data["hand_faces"]:
        closed_face,_ = get_closed_faces(th_faces)
        hand_faces_list.append(closed_face)

    vhacd_exe = "/ghome/l5/ymliu/3rdparty/VHACD_bin/testVHACD"

    sample_infos = geneSampleInfos(fname_lists=data["file_names"],
                                   hand_verts=data["hand_verts"],
                                   hand_faces=hand_faces_list,
                                   object_verts=data["object_verts"],
                                   object_faces=data["object_faces"])
    
    print("<---- Get Sample Info Done ---->")

    
    para_simulate(sample_infos,saved_path, vhacd_exe=vhacd_exe)




