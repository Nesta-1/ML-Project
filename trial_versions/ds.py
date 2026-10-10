import kagglehub

# Download latest version
path = kagglehub.dataset_download("munkialbright/classified-tweets")

print("Path to dataset files:", path)
