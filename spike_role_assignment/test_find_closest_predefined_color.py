from role_assignment import RoleAssigner

role_assigner = RoleAssigner()

input_rgb = [ 78, 94, 128]

print(f"Input RGB color: {input_rgb}")
role_assigner.visualize_color_comparison(input_rgb)