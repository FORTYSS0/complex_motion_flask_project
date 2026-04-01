import plots

print("Available plots:")
for k, title in plots.PLOT_REGISTRY.items():
    print(f"- {k}: {title}")
