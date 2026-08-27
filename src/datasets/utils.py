def load_torchvision_dataset(dataset_cls, **kwargs):
    """Load a local torchvision dataset before attempting a download."""
    try:
        return dataset_cls(download=False, **kwargs)
    except RuntimeError:
        root = kwargs.get("root", "<unknown>")
        print(f"[INFO] {dataset_cls.__name__} not found or invalid at {root}; downloading it.")
        try:
            return dataset_cls(download=True, **kwargs)
        except Exception as download_error:
            raise RuntimeError(
                f"Unable to load {dataset_cls.__name__} from {root} or download it."
            ) from download_error
