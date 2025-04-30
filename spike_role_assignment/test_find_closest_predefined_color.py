from role_assignment import RoleAssigner

role_assigner = RoleAssigner()

input_rgb = [156, 99, 80]

print(f"Input RGB color: {input_rgb}")
role_assigner.visualize_color_comparison(input_rgb)