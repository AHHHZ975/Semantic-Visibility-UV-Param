import sys,os
sys.path.append(os.path.abspath(".."))
from Utils.Funcs import *
from Utils.Workflow import * 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_mesh_path", type=str)    
    parser.add_argument("export_root", type=str)
    parser.add_argument("visibility_awareness", type=str)
    parser.add_argument("semantic_awareness", type=str)
    args = parser.parse_args()

    mesh_name = args.load_mesh_path.split("/")[-1].split(".")[0]
    load_ckpt_path = "trained_model.pth"
    export_folder = args.export_root        

    vertices, normals, faces = load_mesh_model_vfn(args.load_mesh_path)
    num_verts = vertices.shape[0]
    num_faces = faces.shape[0]
    print(f"{num_verts=}, {num_faces=}")
    print(f"start testing on [{mesh_name}] ...")

    if args.visibility_awareness == "True":
        print("########## Visibility-aware UV parameterization ######### ")
        print("########## Computing Ambient Occlusion map for visibility-aware UV parameterization ######### ")
        # Computing per-vertex ambient occlusion map
        # Ambient occlusion is a rendering technique used to calculate the exposure
        # of each point in a surface to ambient lighting. It is usually encoded as 
        # a scalar (normalized between 0 and 1) associated with the vertice of a mesh.
        # [Reference: https://libigl.github.io/libigl-python-bindings/tut-chapter5/]
        ambient_occlusion = compute_vertex_ambient_occlusion(args.load_mesh_path, visualize_ao=False)        
        
        test_visibility_aware_uv_param(vertices, normals, faces, ambient_occlusion, load_ckpt_path, export_folder)

    elif args.semantic_awareness == "True":
        print("########## Semantic-aware UV parameterization ######### ")
        test_semantic_aware_uv_param(vertices, normals, faces, load_ckpt_path, export_folder)

if __name__ == '__main__':
    main()
    print("testing finished.")