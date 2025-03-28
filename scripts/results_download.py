import gdown
import os

if __name__ == "__main__":
    ids = [
        "1c4bGs-GJbK_Ta-66zozZxfv41MIo-qZ-",
        "1-GYBPwSis-jPUbKkOa8vlRMMK9RAKtq2",
        "1k7tUgbIBzT7WjrkvCQ670OVlpO3bKIYM",
    ]
    outputs = [
        "exps.zip",
        "exps_domain_shift.zip",
        "exps_real.zip",
    ]
    for id, output in zip(ids, outputs):
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, output, quiet=False)
        os.system(f"unzip -o {output}")
        os.system(f"rm {output}")
        print(f"Downloaded {output}")
