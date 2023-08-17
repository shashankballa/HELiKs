# Testing v1 without skip_he_ras
# 64x64
O=111000011 N=64   C=64   docker-compose -f mm-docker-compose.yml up
# 128x128
O=111000011 N=128  C=128  docker-compose -f mm-docker-compose.yml up
# 256x256
O=111000011 N=256  C=256  docker-compose -f mm-docker-compose.yml up
# 512x512
O=111000011 N=512  C=512  docker-compose -f mm-docker-compose.yml up
# 1024x1024
O=111000011 N=1024 C=1024 docker-compose -f mm-docker-compose.yml up
# 2048x2048
O=111000011 N=2048 C=2048 docker-compose -f mm-docker-compose.yml up
# 4096x4096
O=111000011 N=4096 C=4096 docker-compose -f mm-docker-compose.yml up
# 8192x8192
O=111000011 N=8192 C=8192 docker-compose -f mm-docker-compose.yml up
# # 16384x16384
# O=111000011 N=16384 C=16384 docker-compose -f mm-docker-compose.yml up
# # 32768x32768
# O=111000011 N=32768 C=32768 docker-compose -f mm-docker-compose.yml up

# Testing v1 with skip_he_ras
# 64x64
O=111000111 N=64   C=64   docker-compose -f mm-docker-compose.yml up
# 128x128
O=111000111 N=128  C=128  docker-compose -f mm-docker-compose.yml up
# 256x256
O=111000111 N=256  C=256  docker-compose -f mm-docker-compose.yml up
# 512x512
O=111000111 N=512  C=512  docker-compose -f mm-docker-compose.yml up
# 1024x1024
O=111000111 N=1024 C=1024 docker-compose -f mm-docker-compose.yml up
# 2048x2048
O=111000111 N=2048 C=2048 docker-compose -f mm-docker-compose.yml up
# 4096x4096
O=111000111 N=4096 C=4096 docker-compose -f mm-docker-compose.yml up
# 8192x8192
O=111000111 N=8192 C=8192 docker-compose -f mm-docker-compose.yml up
# # 16384x16384
# O=111000111 N=16384 C=16384 docker-compose -f mm-docker-compose.yml up
# # 32768x32768
# O=111000111 N=32768 C=32768 docker-compose -f mm-docker-compose.yml up

# Close the docker container
docker-compose -f mm-docker-compose.yml down