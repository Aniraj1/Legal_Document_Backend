MAX_FILE_SIZE_MB = 0.5


def file_size_validate(size):
    file_size_mb = size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False
    return True 


