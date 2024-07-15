import numpy as np
import time

def logsumexp(a, axis=None, keepdims=False):
    """
    To reduce running time I implemented logsumexp myself, the scipy version has too much additional things I don't need
    :param a:
    :param axis:
    :param keepdims:
    :return:
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max[np.isneginf(a_max)] = 0
    output = np.log(np.sum(np.exp(a-a_max),axis=axis, keepdims=keepdims))
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    output += a_max
    return output


def get_non_inf_grid(axis, indices, arr):
    mesh_arr = []
    mesh_a_max = []
    mesh_output = []
    for ax in range(len(indices)):
        if ax == axis or (type(axis) == tuple and ax in axis):
            ax_indices = np.arange(arr.shape[ax])
            mesh_a_max.append(np.array([0]))
            mesh_arr.append(ax_indices)
        else:
            ax_indices, original_index = np.unique(indices[ax], return_index=True)
            ax_indices = ax_indices[np.argsort(original_index)]
            mesh_a_max.append(ax_indices)
            mesh_output.append(ax_indices)
            mesh_arr.append(ax_indices)
    meshgrid_arr = np.meshgrid(*mesh_arr, indexing='ij')
    meshgrid_a_max = np.meshgrid(*mesh_a_max, indexing='ij')
    meshgrid_output = np.meshgrid(*mesh_output, indexing='ij')
    return meshgrid_arr, meshgrid_a_max, meshgrid_output


def logsumexp_(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    indices = np.where(~np.isneginf(a_max))
    a_max[np.isneginf(a_max)] = 0
    meshgrid_arr, meshgrid_a_max, meshgrid_output = get_non_inf_grid(axis, indices, a)
    output = np.full(a.shape[0], -np.inf)
    output[*meshgrid_output] = np.log(
        np.sum(np.exp(a[*meshgrid_arr] - a_max[*meshgrid_a_max]), axis=axis, keepdims=keepdims))

    return 5


def remove_redundant(indices, axis_list):
    indices_to_stack = [indices[j] for j in axis_list]
    stacked_arrays = np.column_stack(indices_to_stack)
    no_repeat = np.unique(stacked_arrays, axis=0)
    for i in range(len(axis_list)):
        indices[axis_list[i]] = no_repeat[:, i]
    return no_repeat


def get_output_shape(a_shape, axis, keepdims):
    new_shape = []
    for i in range(len(a_shape)):
        if i == axis or (type(axis) == tuple and i in axis):
            if keepdims:
                new_shape.append(1)
        else:
            new_shape.append(a_shape[i])
    return tuple(new_shape)


def get_indices(indices, axis, a_shape):
    final_indices = []
    axis_list = []
    for i in range(len(indices)):
        if i == axis:
            final_indices.append(np.arange(a_shape[axis]))
        else:
            final_indices.append(indices[i])
            axis_list.append(i)
    blocks = remove_redundant(final_indices, axis_list)
    return blocks


def get_item(a_shape, axis, blocks):
    row_mask = np.zeros(shape=(blocks.shape[1],))
    size = 1
    for i in range(blocks.shape[0] - 1, -1, -1):
        row_mask += blocks[i,:] * size
        size *= a_shape[axis[i]]
    row_mask = row_mask.astype(np.int64)
    return row_mask



def debugger(a, axis, axis_c, adjusted_axis, output_keepdims, output_nodims, flatten_shape, output_shape, transpose_shape, keepdims=False):
    # tries to calculate mask here but faster
    a_max = np.max(a, axis=axis, keepdims=False)
    a_max_ = np.max(a, axis=axis, keepdims=True)
    a_max_[np.isneginf(a_max_)] = 0
    indices = np.where(a_max != -np.inf)
    blocks = np.vstack(indices)
    flatten_mask = get_item(a.shape, axis_c, blocks)
    flatten_a = a.transpose(transpose_shape).reshape(flatten_shape)
    non_zero_a = flatten_a[..., flatten_mask]
    a_max = np.max(non_zero_a, axis=adjusted_axis)
    non_zero_output = np.log(np.sum(np.exp(non_zero_a - a_max), axis=adjusted_axis))
    output_ = np.log(np.sum(np.exp(a - a_max_), axis=axis, keepdims=keepdims))
    if not keepdims:
        a_max_ = np.squeeze(a_max_, axis=axis)
    non_zero_output += a_max
    output_ += a_max_
    output = np.full(output_shape, -np.inf)
    output[flatten_mask] = non_zero_output
    if keepdims:
        output = output.reshape(output_keepdims)
    else:
        output = output.reshape(output_nodims)
    return output



def sparse_logsumexp(a, axis, axis_c, adjusted_axis, output_keepdims, output_nodims, flatten_shape, output_shape, transpose_shape, keepdims=False):
    # tries to calculate mask here but faster
    a_max = np.max(a, axis=axis, keepdims=False)
    a_max_ = np.max(a, axis=axis, keepdims=True)
    indices = np.where(a_max != -np.inf)
    a_max_[np.isneginf(a_max_)] = 0
    blocks = np.vstack(indices)
    flatten_mask = get_item(a.shape, axis_c, blocks)
    flatten_a = a.transpose(transpose_shape).reshape(flatten_shape)
    non_zero_a = flatten_a[...,flatten_mask]
    a_max = np.max(non_zero_a, axis=adjusted_axis)
    non_zero_output = np.log(np.sum(np.exp(non_zero_a-a_max), axis=adjusted_axis))
    non_zero_output += a_max
    output_ = np.log(np.sum(np.exp(a - a_max_), axis=axis, keepdims=keepdims))
    if not keepdims:
        a_max_ = np.squeeze(a_max_, axis=axis)
    output_ += a_max_
    output = np.full(output_shape,-np.inf)
    output[flatten_mask] = non_zero_output
    if keepdims:
        output = output.reshape(output_keepdims)
    else:
        output = output.reshape(output_nodims)
    return output



# Define the shape of the array
#shape = (512,300,5,96)

# Calculate the total number of elements
#total_elements = np.prod(shape)

# Generate a random array with values between 0 and 1
#x = np.random.rand(*shape)

# Calculate the number of elements to set to -np.inf (50% of total elements)
#num_inf_values = int(total_elements * 0.9999)

# Get random indices to set to -np.inf
#inf_indices = np.random.choice(total_elements, num_inf_values, replace=False)

# Set selected indices to -np.inf
#x.ravel()[inf_indices] = -np.inf
# non_zero_elems = np.where(x != -np.inf)
# blocks = get_indices(non_zero_elems,0,x.shape)
# mask = get_item(shape, (1,2,3), blocks)
#start_time = time.time()
#aaa = debugger(x, (0,1), (2,3), (0,1), (1,1,5,96), (5,96),(512,300,480),(480), (0,1,2,3))
#print(time.time()-start_time)
#start_time = time.time()
#bbb = logsumexp(x, (0,1))
#print(time.time()-start_time)
#a = 5

