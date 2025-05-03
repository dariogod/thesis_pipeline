from role_assignment_refactored import RoleAssigner

role_assigner = RoleAssigner()

input_rgb = [
            67,
            96,
            36
          ]

print(f"Input RGB color: {input_rgb}")
role_assigner.visualize_color_comparison(input_rgb)