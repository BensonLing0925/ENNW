#!/usr/bin/env python3
import os
import gzip
import shutil
import urllib.request

def download_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img":  "t10k-images-idx3-ubyte.gz",
        "test_lbl":  "t10k-labels-idx1-ubyte.gz"
    }
    
    target_dir = os.path.join(os.path.dirname(__file__), "..", "src", "MNIST")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"[*] Created directory: {target_dir}")

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for key, filename in files.items():
        gz_path = os.path.join(target_dir, filename)
        out_path = os.path.join(target_dir, filename.replace(".gz", ""))

        if not os.path.exists(out_path):
            print(f"[+] Downloading {filename}...")
            try:
                urllib.request.urlretrieve(base_url + filename, gz_path)
            except Exception as e:
                print(f"[!] Failed to download {filename}: {e}")
                print("    Please try to download manually if this persists.")
                continue

            print(f"[+] Decompressing {filename}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)
            print(f"[OK] Saved to: {out_path}")
        else:
            print(f"[-] {filename.replace('.gz', '')} already exists, skipping.")

    print("\n[!] MNIST dataset preparation complete in src/MNIST/")

if __name__ == "__main__":
    download_mnist()
