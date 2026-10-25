#!/bin/bash
set -e
DIGEST="52c29de0081cf2fad9fa7a2829d4c46c8d60d0bee73ce710d0e3cb06402de744"
BLOB_DEST="/var/lib/containerd/io.containerd.content.v1.content/blobs/sha256/$DIGEST"

if [ ! -f "$BLOB_DEST" ]; then
    TOKEN=$(curl -s 'https://m.daocloud.io/auth/token?service=m.daocloud.io&scope=repository:quay.io/ascend/vllm-ascend:pull' | jq -r .token)
    echo "Downloading 3.8GB layer with 16 connections..."
    aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=10 --header="Authorization: Bearer $TOKEN" -d /home/models/blobs -o "$DIGEST.blob" "https://m.daocloud.io/v2/quay.io/ascend/vllm-ascend/blobs/sha256:$DIGEST"
    sudo cp "/home/models/blobs/$DIGEST.blob" "$BLOB_DEST"
    sudo chmod 444 "$BLOB_DEST"
fi

echo "Running docker pull (all blobs cached, extracting)..."
sudo docker pull m.daocloud.io/quay.io/ascend/vllm-ascend:v0.23.0.post1-310p
sudo docker tag m.daocloud.io/quay.io/ascend/vllm-ascend:v0.23.0.post1-310p quay.io/ascend/vllm-ascend:v0.23.0.post1-310p
rm -rf /home/models/blobs
echo "FAST_PULL_COMPLETE"
