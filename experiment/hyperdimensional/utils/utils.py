# import math, torch, torchhd

# def create_channels(in_channels, dim, increase, scale_factor):
#     # Find the next power of two greater than in_channels
#     next_power_of_two = 2 ** math.ceil(math.log2(in_channels))

#     # Number of scaling layers is the logarithm base 2 of the scaling factor
#     x = int(math.log2(scale_factor))

#     # Generate down_channels
#     current_channels = next_power_of_two
#     down_channels = [(in_channels, current_channels, 1)]  # Initial jump to the next power of two with no scaling
#     steps_needed = (dim - current_channels) // increase + (1 if (dim - current_channels) % increase != 0 else 0)

#     for i in range(1, steps_needed + 1):
#         next_channels = current_channels + increase
#         if i > steps_needed - x:  # Apply scaling only in the last x layers
#             scale = 2
#         else:
#             scale = 1
#         down_channels.append((current_channels, min(next_channels, dim), scale))
#         current_channels = next_channels

#     if current_channels < dim:
#         down_channels.append((current_channels, dim, 2))

#     # Generate up_channels to exactly reverse down_channels
#     up_channels = []
#     for i in range(len(down_channels)-1, 0, -1):
#         down = down_channels[i]
#         up = down_channels[i-1]
#         scale = down[2]  # Use the same scale as down_channels for symmetry
#         up_channels.append((down[1], up[1], scale))

#     # Ensure we start and end at the same channel sizes as down_channels
#     up_channels.append((down_channels[0][1], down_channels[0][0], down_channels[0][2]))

#     return down_channels, up_channels

# def create_channelsV2(in_channels, dim, scale_factor):
#     # Find the next power of two greater than in_channels
#     next_power_of_two = 2 ** math.ceil(math.log2(in_channels))

#     # Number of scaling layers is the logarithm base 2 of the scale_factor
#     x = int(math.log2(scale_factor))

#     # Generate down_channels
#     current_channels = next_power_of_two
#     down_channels = [(in_channels, current_channels, 1)]  # Initial jump to the next power of two with no scaling

#     steps_needed = math.ceil(math.log2(dim / current_channels))  # Calculate steps needed based on final dimension
#     for i in range(steps_needed):
#         next_channels = current_channels * 2
#         if next_channels > dim:
#             next_channels = dim
#         # Apply scale of 2 only to the last x layers
#         scale = 2 if steps_needed - i <= x else 1
#         down_channels.append((current_channels, next_channels, scale))
#         current_channels = next_channels
#         if current_channels == dim:
#             break

#     # Generate up_channels to exactly reverse down_channels
#     up_channels = []
#     for i in range(len(down_channels)-1, 0, -1):
#         down = down_channels[i]
#         up = down_channels[i-1]
#         scale = down[2]  # Use the same scale as down_channels for symmetry
#         up_channels.append((down[1], up[1], scale))

#     # Ensure we start and end at the same channel sizes as down_channels
#     up_channels.append((down_channels[0][1], down_channels[0][0], down_channels[0][2]))

#     return down_channels, up_channels

# def sign(tensor):
#     return torch.where(tensor < 0, torch.tensor(-1.0), torch.tensor(1.0))

# def review_sims(tensor, dim):
#     sims = torchhd.cosine_similarity(tensor, tensor).mean(dim=dim)
#     print(f"Review Max Sim: {sims[torch.argmax(sims).item()]}")
#     print(f"Review Min Sim: {sims[torch.argmin(sims).item()]}")

# def scale_tensor(x, old_min, old_max, new_min, new_max):
#     return new_min + ((new_max - new_min) * (x - old_min) / (old_max - old_min))