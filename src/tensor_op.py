import torch
import numpy as np

epsilon = 0.00001

def TensorUnfold(input_tensor, k):
    shape_tensor = list(input_tensor.size())
    num_dim = len(shape_tensor)
    permute_order = [k] + np.delete(range(num_dim), k).tolist()
    middle_result = input_tensor.permute(*permute_order)
    shape_middle = list(middle_result.size())
    result = middle_result.view([shape_middle[0], np.prod(shape_middle[1:])])
    return result

def TensorProduct(tensor1, tensor2, axes=(0, 0)):
    shape1 = list(tensor1.size())
    shape2 = list(tensor2.size())
    shape_out = np.delete(shape1, axes[0]).tolist() + np.delete(shape2, axes[1]).tolist()
    result = torch.matmul(torch.t(TensorUnfold(tensor1, axes[0])), TensorUnfold(tensor2, axes[1]))
    return result.resize_(shape_out)

def UpdateCov(matrix_out, tensor1, tensor2):
    size_dim0 = matrix_out.size(0)
    size_dim1 = matrix_out.size(1)
    size_dim2 = matrix_out.size(2)
    middle_result = torch.mm(matrix_out.view(-1, size_dim2), tensor2)
    middle_result1 = torch.mm(middle_result.view(size_dim0, size_dim1, size_dim2).permute(0, 2, 1).contiguous().view(-1, size_dim1), tensor1).view(size_dim0, size_dim2, size_dim1).permute(0, 2, 1).contiguous()
    final_result = torch.mm(middle_result1.view(size_dim0, -1), torch.t(middle_result1.view(size_dim0, -1)))
    return torch.inverse(final_result + epsilon * torch.eye(final_result.size(0)).cuda())   

def MultiTaskLoss(weight_matrix, tensor1, tensor2, tensor3):
    size_dim0 = weight_matrix.size(0)
    size_dim1 = weight_matrix.size(1)
    size_dim2 = weight_matrix.size(2)
    middle_result = torch.mm(tensor3, weight_matrix.permute(2, 0, 1).contiguous().view(size_dim2, -1))
    middle_result = torch.mm(tensor2, middle_result.view(size_dim2, size_dim0, size_dim1).permute(2, 1, 0).contiguous().view(size_dim1, -1))
    middle_result = torch.mm(tensor1, middle_result.view(size_dim1, size_dim0, size_dim2).permute(1, 0, 2).contiguous().view(size_dim0, -1)).view(-1, 1)
    return torch.mm(weight_matrix.view(1, -1), middle_result).view(1)
    
if __name__ == "__main__":
    tensor1 = torch.randn(2,3,4)
    tensor2 = torch.randn(3,4,5)
    result = TensorProduct(tensor1, tensor2, (1, 0))
    print(tensor1)
    print(tensor2)
    print(result)
