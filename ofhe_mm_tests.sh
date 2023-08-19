# Close old docker container
O=1111111111 N=32768 C=32768 docker-compose -f ofhe_mm_docker-compose.yml down

# Testing 64x64 
# CF2
O=1100000110 N=64   C=64   docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=64   C=64   docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=64   C=64   docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 128x128
# CF2
O=1100000110 N=128  C=128  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=128  C=128  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=128  C=128  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 256x256
# CF2
O=1100000110 N=256  C=256  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=256  C=256  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=256  C=256  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 512x512
# CF2
O=1100000110 N=512  C=512  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=512  C=512  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=512  C=512  docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 1024x1024
# CF2
O=1100000110 N=1024 C=1024 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=1024 C=1024 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=1024 C=1024 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 2048x2048
# CF2
O=1100000110 N=2048 C=2048 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=2048 C=2048 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=2048 C=2048 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 4096x4096
# CF2
O=1100000110 N=4096 C=4096 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=4096 C=4096 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=4096 C=4096 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 8192x8192
# CF2
O=1100000110 N=8192 C=8192 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=8192 C=8192 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=8192 C=8192 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 16384x16384
# CF2
O=1100000110 N=16384 C=16384 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=16384 C=16384 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=16384 C=16384 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Testing 32768x32768
# CF2
O=1100000110 N=32768 C=32768 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v1
O=1110001110 N=32768 C=32768 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans
# v2
O=1111001111 N=32768 C=32768 docker-compose -f ofhe_mm_docker-compose.yml up --remove-orphans

# Close new docker container
O=1111111111 N=32768 C=32768 docker-compose -f ofhe_mm_docker-compose.yml down