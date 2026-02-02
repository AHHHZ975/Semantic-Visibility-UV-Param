import sys,os
sys.path.append(os.path.abspath(".."))
from Utils.Funcs import *
from Utils.Workflow import *

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument("load_mesh_path", type=str)
    parser.add_argument("export_root", type=str)    
    parser.add_argument("num_iter", type=int)
    parser.add_argument("visibility_awareness", type=str)
    parser.add_argument("semantic_awareness", type=str)
    args = parser.parse_args()
    
    mesh_name = args.load_mesh_path.split("/")[-1].split(".")[0]
    export_folder = os.path.join(args.export_root, mesh_name+"_"+str(1))
    os.makedirs(export_folder, exist_ok=True)

    # Loading the mesh object and extract required data including
    # vertices, triangle faces, and normal vectors.
    vertices, normals, faces, edge_index = load_mesh_and_edges(args.load_mesh_path)
    num_verts = vertices.shape[0]
    num_faces = faces.shape[0]

    print(f"{num_verts=}, {num_faces=}")
    print(f"start training on [{mesh_name}] ...")

    if args.visibility_awareness == "True":
        print("########## Computing Ambient Occlusion map for visibility-aware UV parameterization ######### ")
        # Computing per-vertex ambient occlusion map
        # Ambient occlusion is a rendering technique used to calculate the exposure
        # of each point in a surface to ambient lighting. It is usually encoded as 
        # a scalar (normalized between 0 and 1) associated with the vertice of a mesh.
        # [Reference: https://libigl.github.io/libigl-python-bindings/tut-chapter5/]
        ambient_occlusion = compute_vertex_ambient_occlusion(args.load_mesh_path, visualize_ao=True)
        ambient_occlusion = torch.from_numpy(ambient_occlusion.astype(np.float32)).cuda()  # shape [V]

        # Train our visibility-aware UV parameterization pipeline               
        train_visibility_aware_uv_param(vertices, normals, faces, ambient_occlusion, args.num_iter, export_folder)

    elif args.semantic_awareness == "True":
        # Train our semantic-aware UV parameterization pipeline
        train_semantic_aware_uv_param(vertices, normals, faces, args.num_iter, export_folder)


if __name__ == '__main__':
    main()
    print("training finished.")
