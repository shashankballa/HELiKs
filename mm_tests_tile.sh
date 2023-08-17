# Testing v2 (16000 x 500)
O=1111001111 N=16000 C=500 docker-compose -f mm-docker-compose.yml up

# Testing v2 (16000 x 500)
O=1111001110 N=16000 C=500 docker-compose -f mm-docker-compose.yml up

# Testing v2 (500 x 16000)
O=1111001111 N=500 C=16000 docker-compose -f mm-docker-compose.yml up

# Testing v2 (500 x 16000)
O=1111001110 N=500 C=16000 docker-compose -f mm-docker-compose.yml up

# Testing v2 (16000 x 16000)
O=1111001111 N=16000 C=16000 docker-compose -f mm-docker-compose.yml up

# Testing v2 (16000 x 16000)
O=1111001110 N=16000 C=16000 docker-compose -f mm-docker-compose.yml up

# Close the docker container
docker-compose -f mm-docker-compose.yml down