def convert_bytes_to_formatted_size(byte_size):
    file_type = ["Bytes", "KB", "MB", "GB"]
    i = 0
    size = byte_size
    while size > 900:
        size /= 1024
        i += 1
    exact_size = f"{round(size, 2)} {file_type[i]}"
    return exact_size