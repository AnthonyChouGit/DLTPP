import torch

def embed_rounding(input_embed,  output_embed):
    '''
        input_embed: (num_mark, embed_size)
        output_embed: (..., embed_size)

    Returns:
        (..., embed_size)
    '''
    input_embed_norm_square = (input_embed**2).sum(-1) # (num_marks)
    output_embed_norm_square = (output_embed**2).sum(-1) # (...)
    # (num_mark, embed_size)*(..., embed_size, 1)=(..., num_mark, 1)
    cross_term = 2*torch.matmul(input_embed, output_embed.unsqueeze(-1)).squeeze(-1) # (..., num_mark)
    dist = input_embed_norm_square + output_embed_norm_square.unsqueeze(-1) - cross_term
    inds = torch.argmin(dist, dim=-1) # (...)
    round_embed = input_embed[inds]
    return round_embed
