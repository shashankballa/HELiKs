H=230 FH=7 CI=3    CO=64   S=2 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=56  FH=1 CI=64   CO=256  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=56  FH=1 CI=64   CO=64   S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=56  FH=3 CI=64   CO=64   S=1 P=1 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=56  FH=1 CI=256  CO=64   S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=56  FH=1 CI=256  CO=512  S=2 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=56  FH=1 CI=256  CO=128  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=58  FH=3 CI=128  CO=128  S=2 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=28  FH=1 CI=128  CO=512  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=28  FH=1 CI=512  CO=128  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=28  FH=3 CI=128  CO=128  S=1 P=1 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=28  FH=1 CI=512  CO=1024 S=2 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=28  FH=1 CI=512  CO=256  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=30  FH=3 CI=256  CO=256  S=2 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=14  FH=1 CI=256  CO=1024 S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=14  FH=1 CI=1024 CO=256  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=14  FH=3 CI=256  CO=256  S=1 P=1 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=14  FH=1 CI=1024 CO=2048 S=2 P=0 O=0111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=14  FH=1 CI=1024 CO=512  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=16  FH=3 CI=512  CO=512  S=2 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=7   FH=1 CI=512  CO=2048 S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=7   FH=1 CI=2048 CO=512  S=1 P=0 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans
H=7   FH=3 CI=512  CO=512  S=1 P=1 O=1111 docker-compose -f seal_cv_docker-compose.yml up --remove-orphans

H=7   FH=3 CI=512  CO=512  S=1 P=1 O=1111 docker-compose -f seal_cv_docker-compose.yml down