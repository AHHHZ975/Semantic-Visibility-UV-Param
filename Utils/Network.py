import os, sys
sys.path.append(os.path.abspath('..'))
from Utils.Funcs import *
from typing import List

class PWE(nn.Module):
    # point-wise embedding: Conv_with_bias (without BN) + LeakyReLU
    def __init__(self, C_in, C_out, neg_slope=None):
        super(PWE, self).__init__()
        self.neg_slope = neg_slope
        self.conv = nn.Conv1d(C_in, C_out, 1, bias=True)
    def forward(self, P_in):
        # P_in: [B, N, C_in]
        # B is the "batch size"
        # N is the "number of points"
        # C_in is the "feature dimension"
        P_in = P_in.permute(0, 2, 1).contiguous() # [B, C_in, N]
        P_out = self.conv(P_in) # [B, C_out, N]
        if self.neg_slope is not None:
            P_out = F.leaky_relu(P_out, self.neg_slope, True) # [B, C_out, N]
        P_out = P_out.permute(0, 2, 1).contiguous() # [B, N, C_out]
        return P_out # [B, N, C_out]

##############################################################################################################
############################################### Global FlexPara: FAM #########################################
##############################################################################################################
# Within the Cut-Net (Cutting), "self.mlp_1" and "slef.mlp_2" are three-layer
# MLPs with channels [3, 512, 512, 64] and [67, 512, 512, 3], respectively.
# MY GUESS (don't take it for granted) is that the Cut-Net is applied to 
# the 3D mesh surface and not in the UV space and that's why it outputs 
# a tensor with the last dimension of 3 and my intuition is that this tensor
# contains a series of 3D points on the mesh surface that shows the cutting curve.

# The cutting network Mc(·) (Cut-Net) operates in the 3D space to derive appropriate cutting seams
# on the target 3D surface, transforming the original geometric structure into an open and more
# developable surface manifold.
class Cutting(nn.Module):
    def __init__(self):
        super(Cutting, self).__init__()
        hidden_dim = 64
        self.mlp_1 = nn.Sequential(
            PWE(3, 512, 0.01), 
            PWE(512, 512, 0.01), 
            PWE(512, hidden_dim, None)
        )
        self.mlp_2 = nn.Sequential(
            PWE(hidden_dim+3, 512, 0.01), 
            PWE(512, 512, 0.01), 
            PWE(512, 3, None)
        )
    
    # To construct cycle mapping, we sequentially apply the CutNet and 
    # the Unwrap-Net for the flattening of "P_hat" ("X" here). Specifically, 
    # we transform "P_hat" ("X" here) to an open 3D manifold "P_hat_cut", which
    # should be much more developable than "P_hat", by learning pointwise offsets
    # as presented in the equation (3) in the paper.
    def forward(self, X):
        # X: [B, N, 3]
        Xo = self.mlp_2(torch.cat((X, self.mlp_1(X)), dim=-1)) # [B, N, 3] -> This is the offse we're talking about in the equation (3) of the paper.
        Xc = X + Xo
        return Xc # This is the "P_hat_cut" or "P_cut" in the paper contains all the cutting seam points.



# Modified Cutting Network with Seam Branch
# This functions exactly the same as our vanilla
# "Cutting" except it also has
class Cutting_VisibilityAware(nn.Module):
    """
    Cut-Net learns:
      1) an offset field to open the mesh (Xc = X + offsets)
      2) a per-point seam score, used in visibility loss
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 1) feature extractor
        self.mlp_feat = nn.Sequential(
            PWE(3, 512, neg_slope=0.01),
            PWE(512, 512, neg_slope=0.01),
            PWE(512, hidden_dim)
        )
        # 2) offset regressor (3D deformation)
        self.mlp_offset = nn.Sequential(
            PWE(hidden_dim + 3, 512, neg_slope=0.01),
            PWE(512, 512, neg_slope=0.01),
            PWE(512, 3)
        )
        # 3) seam score head (logit → prob)
        self.mlp_seam = nn.Sequential(
            PWE(hidden_dim, 512, neg_slope=0.01),
            PWE(512, 512, neg_slope=0.01),
            PWE(512, 1)
        )

    def forward(self, X):
        """
        Args:
          X: Tensor[B, N, 3], input points
        Returns:
          Xc: Tensor[B, N, 3], cut/opened points
          seam_prob_v: Tensor[B, N], per-vertex seam probability in [0,1]
        """
        B, N, _ = X.shape

        # 1) Extract per-point features
        f = self.mlp_feat(X)                 # [B, N, hidden_dim]

        # 2) predict offsets and apply them
        offset = self.mlp_offset(torch.cat([X, f], dim=-1))  # [B, N, 3]
        Xc = X + offset                      # opened mesh
        
        # 3) predict seam logits and convert to probabilities
        seam_logit_v = self.mlp_seam(f).squeeze(-1)  # [B, N]
        seam_prob_v  = torch.sigmoid(seam_logit_v)   # [B, N]

        return Xc, seam_prob_v



# Within the Unwrap-Net, "self.mlp" is a three-layer MLP 
# with channels [3, 512, 512, 2].
# The unwrapping network Mu(·) (Unwrap-Net) smoothly
# flattens 3D surface points onto the 2D parameter domain.
class Unwrapping(nn.Module):
    def __init__(self):
        super(Unwrapping, self).__init__()
        self.cut = Cutting()
        self.mlp = nn.Sequential(PWE(3, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 2, None))

    # The "points_3d" ("P_hat") has the shape of [1, N, 3] and contains resultant (reconstructed) points from after applying the Wrap-Net which converts
    # 2D initial UV coordinates (G) to "P_hat". So, to construct cycle mapping, we sequentially apply the CutNet and
    # the Unwrap-Net for flattening "P_hat". 
    def forward(self, points_3d):

        # Specifically, we transform "points_3d" ("P_hat" or "P", depending on we are in the 3D-2D-3D or 2D-3D-2D branch)
        # to an open 3D manifold "P_hat_cut" or "P_cut", (depending on we are in the 3D-2D-3D or 2D-3D-2D branch), 
        # should be much more developable than "P_hat", by learning pointwise offsets
        # as presented in the equations (3) and (5) in the paper.
        dfm = self.cut(points_3d)

        # After that we perform 3D-to-2D unwrapping as presented in the equation (4) and (6) in the paper.
        unwrapped_points_2d = self.mlp(dfm) # [B, N, 2]

        return dfm, unwrapped_points_2d



# Within the Unwrap-Net, "self.mlp" is a three-layer MLP 
# with channels [3, 512, 512, 2].
# The unwrapping network Mu(·) (Unwrap-Net) smoothly
# flattens 3D surface points onto the 2D parameter domain.
class Unwrapping_VisibilityAware(nn.Module):
    def __init__(self):
        super(Unwrapping_VisibilityAware, self).__init__()
        self.cut = Cutting_VisibilityAware()
        self.mlp = nn.Sequential(PWE(3, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 2, None))

    # The "points_3d" ("P_hat") has the shape of [1, N, 3] and contains resultant (reconstructed) points from after applying the Wrap-Net which converts
    # 2D initial UV coordinates (G) to "P_hat". So, to construct cycle mapping, we sequentially apply the CutNet and
    # the Unwrap-Net for flattening "P_hat". 
    def forward(self, points_3d):

        # Specifically, we transform "points_3d" ("P_hat" or "P", depending on we are in the 3D-2D-3D or 2D-3D-2D branch)
        # to an open 3D manifold "P_hat_cut" or "P_cut", (depending on we are in the 3D-2D-3D or 2D-3D-2D branch), 
        # should be much more developable than "P_hat", by learning pointwise offsets
        # as presented in the equations (3) and (5) in the paper.
        dfm, self.seam_prob_v = self.cut(points_3d)

        # After that we perform 3D-to-2D unwrapping as presented in the equation (4) and (6) in the paper.
        unwrapped_points_2d = self.mlp(dfm) # [B, N, 2]

        return dfm, unwrapped_points_2d, self.seam_prob_v



# Within the Deform-Net (GridDeforming), "slef.mlp_1" and "slef.mlp_2" are four-layer 
# MLPs with channels [2, 512, 512, 512, 64] and [66, 512, 512, 512, 2].
# The deforming network Md(·) (Deform-Net) takes as input a set of uniform grid coordinates
# located at a predefined 2D lattice. The initial grids will be adaptively deformed to produce
# potentially-optimal UV coordinates
class GridDeforming(nn.Module):
    def __init__(self):
        super(GridDeforming, self).__init__()
        hidden_dim = 64
        self.mlp_1 = nn.Sequential(PWE(2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, hidden_dim, None))
        self.mlp_2 = nn.Sequential(PWE(hidden_dim+2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 2, None))
    
    # Treating G (or "points_2d" here) as the initial 2D parameter domain, we tend to explicitly deform it to produce potentially optimal
    # UV coordinates through the Deform-Net. Concretely, the UV space deformation is implemented as an offset-driven
    # coordinate updating process, which can be formulated as the equation (1) in the paper. Then, The learned offsets
    # are point-wisely added to the initial 2D grid points to produce the resulting Q.
    def forward(self, points_2d):
        # points_2d: [B, N, 2]
        offsets_2d = self.mlp_2(torch.cat((points_2d, self.mlp_1(points_2d)), dim=-1)) # [B, N, 2]
        deformed_points_2d = points_2d + offsets_2d # [B, N, 2]
        return deformed_points_2d



# Within the Wrap-Net (Wrapping), "slef.mlp_1" and "slef.mlp_2" are four-layer 
# MLPs with channels [2, 512, 512, 512, 64] and [66, 512, 512, 512, 6]. 
# The "6" in the last dimension includes 3 for 3D cooridnates for each vertex on the mesh
# and 3 for normal vectors at each vertex.
# The wrapping network Mw(·) (Wrap-Net) performs 2D-to-3D mapping. Intuitively, 
# the potentially-optimal planar UV points will be smoothly folded to approximate
# the target 3D surface.
class Wrapping(nn.Module):
    def __init__(self):
        super(Wrapping, self).__init__()
        hidden_dim = 64
        # "self.mlp_1" is used to encode the 2D UV coordinates and then is concatenated with each 2D UV coordinate.
        self.mlp_1 = nn.Sequential(PWE(2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, hidden_dim, None))
        self.mlp_2 = nn.Sequential(PWE(hidden_dim+2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 6, None))
    
    # After UV space deformation, we feed "Q_hat" ("points_2d") into the Wrap-Net to generate a 3D point cloud "P_hat" ("wrapped_points"), whose underlying
    # surface should approximate the original geometric structure of "P". In the meantime, the Wrap-Net is configured with another
    # 3 output channels to produce point-wise normals "P_hat_n" ("wrapped_normals"). The whole wrapping process can be formulated as the equation (2) in the paper.
    # The intuition here is that, given 2D deformed UV coordinates as input, we want to train an MLP such that it produces 3D points and their normal vectors very close to the
    # original surface 3D points and normals.
    def forward(self, points_2d):
        # points_2d: [B, N, 2]
        wrapped_6d = self.mlp_2(torch.cat((points_2d, self.mlp_1(points_2d)), dim=-1)) # [B, N, 6]
        wrapped_points = wrapped_6d[:, :, 0:3] # [B, N, 3]
        wrapped_normals = wrapped_6d[:, :, 3:6] # [B, N, 3]
        return wrapped_points, wrapped_normals

class FlexParaGlobal(nn.Module):
    def __init__(self):
        super(FlexParaGlobal, self).__init__()
        self.unwrapping = Unwrapping()
        self.grid_deforming = GridDeforming()
        self.wrapping = Wrapping()
    
    # "P" has the shape of [1, N, 3] and contains sampled 3D point coordinates on the mesh surface (not their embedding) at the current training iteration
    # "G" has the shape of [1, M, 2] and is an initial UV coordinates - a set of planar grid coordinates uniformly sampled from a pre-defined grid within [−1, 1]^2. 
    # (Look at the Figure 2 in the FlexPara paper to understand what these two are)
    def forward(self, G , P):

        #### [3D -> 2D -> 3D] cycle mapping
        # Look at the Figure 2 (b) in the paper. This includes three steps below:
        # 1- Performing the cutting operation on the input point cloud (P) using the Cut-Net
        # 2- Performing the unwraping operation using the Unwrap-Net
        # 3- Performing the wrapping operation using teh Wrap-Net.
        
        # "P_opened" is the "P_cut" in the paper which is the output after applying the cutting operation
        # "Q" is the UV parameterization (coordinates) after applying cutting-unwrapping operations.
        # This is going to be the actual UV parameterization we will need after the training process.
        # Equations (5) and (6) in the paper.
        P_opened, Q = self.unwrapping(P)

        # "P_cycle" is the reconstructed point clouds after unwrapping-cutting-wrapping operations.
        # "P_cycle_n" is the reconstructed normal vectors after unwrapping-cutting-wrapping operations.
        # This is the equation (7) in the paper.
        P_cycle, P_cycle_n = self.wrapping(Q)

        
        #### [2D -> 3D -> 2D] cycle mapping
        # Look at the Figure 2 (a) in the paper. This includes three steps below:
        # 1- Performing the deforming operation on the initial UV grid (G) using the Deform-Net
        # 2- Performing the wrapping operation using the Wrap-Net.
        # 3- Performing the cutting operation using the Cut-Net.
        # 4- Performing the unwrapping operation using Unwrap-Net.

        # "Q_hat" is the deformed UV coordinates after applying the deforming operation on the initial UV grid (G)
        # Treating G as the initial 2D parameter domain, we tend to explicitly deform it to produce potentially optimal
        # UV coordinates through the Deform-Net. Concretely, the UV space deformation is implemented as an offset-driven
        # coordinate updating process, which can be formulated as the equation (1) in the paper. Then, The learned offsets
        # are point-wisely added to the initial 2D grid points to produce the resulting Q.
        Q_hat = self.grid_deforming(G)

        # After UV space deformation, we feed "Q_hat" into the Wrap-Net to generate a 3D point cloud "P_hat", whose underlying
        # surface should approximate the original geometric structure of "P". In the meantime, the Wrap-Net is configured with another
        # 3 output channels to produce point-wise normals "P_hat_n". The whole wrapping process can be formulated as the equation (2) in the paper.
        # "P_hat" is the 3D points resulted from the deformed UV coordinated (Q_hat) after applying the wrapping operation
        # "P_hat_n" is the 3D normal vectors resulted from the deformed UV coordinated (Q_hat) after applying the wrapping operation 
        P_hat, P_hat_n = self.wrapping(Q_hat)

        # "P_hat_opened" is the "P_hat_cut" in the paper in the Figure 2 (a) and is an open and more developable surface manifold than the original geometric structure.
        # "Q_hat_cycle" is the 2D UV coordinates resulted from unwrapping the "P_hat" after applying the unwrapping operation.
        P_hat_opened, Q_hat_cycle = self.unwrapping(P_hat)

        return P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle
    

class FlexParaGlobal_VisibilityAware(nn.Module):
    def __init__(self):
        super(FlexParaGlobal_VisibilityAware, self).__init__()
        self.unwrapping = Unwrapping_VisibilityAware()
        self.grid_deforming = GridDeforming()
        self.wrapping = Wrapping()

    # "P" has the shape of [1, N, 3] and contains sampled 3D point coordinates on the mesh surface (not their embedding) at the current training iteration
    # "G" has the shape of [1, M, 2] and is an initial UV coordinates - a set of planar grid coordinates uniformly sampled from a pre-defined grid within [−1, 1]^2. 
    # (Look at the Figure 2 in the FlexPara paper to understand what these two are)
    def forward(self, G , P):

        #### [3D -> 2D -> 3D] cycle mapping
        # Look at the Figure 2 (b) in the paper. This includes three steps below:
        # 1- Performing the cutting operation on the input point cloud (P) using the Cut-Net
        # 2- Performing the unwraping operation using the Unwrap-Net
        # 3- Performing the wrapping operation using teh Wrap-Net.
        
        # "P_opened" is the "P_cut" in the paper which is the output after applying the cutting operation
        # "Q" is the UV parameterization (coordinates) after applying cutting-unwrapping operations.
        # "seam_prob_v" is the probability of existence of seam on each edge on 3D object.
        # This is going to be the actual UV parameterization we will need after the training process.
        # Equations (5) and (6) in the paper.
        P_opened, Q, self.seam_prob_v = self.unwrapping(P)        

        # "P_cycle" is the reconstructed point clouds after unwrapping-cutting-wrapping operations.
        # "P_cycle_n" is the reconstructed normal vectors after unwrapping-cutting-wrapping operations.
        # This is the equation (7) in the paper.
        P_cycle, P_cycle_n = self.wrapping(Q)

        
        #### [2D -> 3D -> 2D] cycle mapping
        # Look at the Figure 2 (a) in the paper. This includes three steps below:
        # 1- Performing the deforming operation on the initial UV grid (G) using the Deform-Net
        # 2- Performing the wrapping operation using the Wrap-Net.
        # 3- Performing the cutting operation using the Cut-Net.
        # 4- Performing the unwrapping operation using Unwrap-Net.

        # "Q_hat" is the deformed UV coordinates after applying the deforming operation on the initial UV grid (G)
        # Treating G as the initial 2D parameter domain, we tend to explicitly deform it to produce potentially optimal
        # UV coordinates through the Deform-Net. Concretely, the UV space deformation is implemented as an offset-driven
        # coordinate updating process, which can be formulated as the equation (1) in the paper. Then, The learned offsets
        # are point-wisely added to the initial 2D grid points to produce the resulting Q.
        Q_hat = self.grid_deforming(G)

        # After UV space deformation, we feed "Q_hat" into the Wrap-Net to generate a 3D point cloud "P_hat", whose underlying
        # surface should approximate the original geometric structure of "P". In the meantime, the Wrap-Net is configured with another
        # 3 output channels to produce point-wise normals "P_hat_n". The whole wrapping process can be formulated as the equation (2) in the paper.
        # "P_hat" is the 3D points resulted from the deformed UV coordinated (Q_hat) after applying the wrapping operation
        # "P_hat_n" is the 3D normal vectors resulted from the deformed UV coordinated (Q_hat) after applying the wrapping operation 
        P_hat, P_hat_n = self.wrapping(Q_hat)

        # "P_hat_opened" is the "P_hat_cut" in the paper in the Figure 2 (a) and is an open and more developable surface manifold than the original geometric structure.
        # "Q_hat_cycle" is the 2D UV coordinates resulted from unwrapping the "P_hat" after applying the unwrapping operation.
        P_hat_opened, Q_hat_cycle, _ = self.unwrapping(P_hat)

        # print(self.seam_prob_v.shape)

        return P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle, self.seam_prob_v
 

##############################################################################################################
############################################### Multi-Chart FlexPara #########################################
##############################################################################################################


# Within the Assign-Net (AssignNet), "self.mlp" is a two layer MLPs with channels [512, 512, K].
# The "K" is the number of charts (num_charts) here.
# This network explicitly measures the probability of each 3D point belonging to all K charts.
class AssignNet(nn.Module):
    def __init__(self, num_charts):
        super(AssignNet, self,).__init__()
        self.mlp = nn.Sequential(PWE(512, 512, 0.01),PWE(512, num_charts, None))
    def forward(self, fusion_tensor):
        # "fusion_tensor" is the embedding tensor of 3D vertices with the shape of [B, N, 512]
        charts_prob = self.mlp(fusion_tensor) # [B, N, num_charts]
        charts_prob = F.softmax(charts_prob, dim=-1) # This is the probability assigend to each input vertex point
                                                     # This determines which point on the 3D surface belongs to which chart/atlas.
        return charts_prob

# The embedding module is a two-layer MLPs with channels [3, 512, 512].
# Converts each point (x,y,z) to a 512-dimensional embedding tensor
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.mlp = nn.Sequential(PWE(3, 512, 0.01), PWE(512, 512, 0.01))
    def forward(self, P):
        # P: [B, N, 3]        
        fusion_tensor = self.mlp(P) # [B, N, 512]
        return fusion_tensor

# Withing the Unwrap-Net (UnwrappingEmbedding), the channels are changed to
# [512, 512, 512, 2] with a high dimensional feature vector as input.
class UnwrappingEmbedding(nn.Module):
    def __init__(self):
        super(UnwrappingEmbedding, self).__init__()
        self.mlp = nn.Sequential(PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 2, None))
    def forward(self, fusion_tensor):
        # "fusion_tensor" is the embedding tensor of 3D vertices with the shape of [B, N, 512]
        unwrapped_points_2d = self.mlp(fusion_tensor) # [B, N, 2]
        return fusion_tensor, unwrapped_points_2d



# To precisely deduce point-wise UV mappings of P, they develop a
# 3D -> 2D -> 3D cycle mapping branch, which maps P (the vertices on the input mesh)
# into the 2D UV space and mapps them back into 3D space.
# In this way, when the training process is finished, the networks are adapted
# to P to point-wisely produce UV coordinates.
class SingleBranchCycleMapping(nn.Module):
    def __init__(self):
        super(SingleBranchCycleMapping, self).__init__()
        self.unwrapping = UnwrappingEmbedding()
        self.wrapping = Wrapping()
    def forward(self, fusion_tensor):
        #### [3D -> 2D -> 3D] cycle mapping
        # "fusion_tensor" is the embedding tensor of 3D vertices with the shape of [B, N, 512]
        P_opened, Q = self.unwrapping(fusion_tensor) # Maps every 3D vertex into a 2D coordinate in UV space (Q)
                                                     # "P_opened" is exactly same as "fusion_tensor"
        P_cycle, P_cycle_n = self.wrapping(Q) # This one takes the 2D unwrapped UV points and returns 3D vertices and 3D normal vectors

        # "P_opened": The embedding of each 3D vertex
        # "Q": The actual unwrapped 2D UV coordinates. This would be our final UV parameterization after training.
        # "P_cycle": This is the prediced vertices after performing the 3D-2D-3D process.
        # "P_cycle_n": This is the predicted normal vectors after performing the 3D-2D-3D process.
        return P_opened, Q, P_cycle, P_cycle_n

class FlexParaMultiChart(nn.Module):
    def __init__(self, num_charts):
        super(FlexParaMultiChart, self).__init__()
        self.charts = AssignNet(num_charts)                
        self.num_charts = num_charts
        self.cycle_mappings = nn.ModuleList([SingleBranchCycleMapping() for _ in range(num_charts)])
        self.embedding = Embedding()

    def forward(self, P):
        
        # We start by embedding the original point set P into
        # the latent space to produce 512-dimensional feature vectors.
        embedding_tensor = self.embedding(P)

        # The "self.charts" is the Assign-Net MLP model in the paper,
        # that takes the embedding of the 3D coordinates of the vertices
        # of the input mesh object and determines which vertex point belongs
        # to which chart and returns a probability value that shows.
        # The Assign-Net is naturally driven to update the output chart assignment
        # scores, such that each surface chart is as developable as possible.
        # To achieve adaptively-learnable chart assignment, we apply the Assign-Net
        # to explicitly generate K assignment scores for each 3D point, 
        charts_prob = self.charts(embedding_tensor)

        # Then for the number of charts (defined by the user), 
        # Do the a single 3D -> 2D -> 3D mapping to make sure
        # that the UV parameterization is bijective.
        results = []
        for i in range(self.num_charts):
            results.append(self.cycle_mappings[i](embedding_tensor))
        return results, charts_prob
