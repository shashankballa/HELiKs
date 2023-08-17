#Build SCI
cd $ROOT/SCI
mkdir -p build
cd build

if [[ "$NO_REVEAL_OUTPUT" == "NO_REVEAL_OUTPUT" ]]; then
	cmake -DCMAKE_INSTALL_PREFIX=./install ../ -DNO_REVEAL_OUTPUT=ON
else
  cmake -DCMAKE_INSTALL_PREFIX=./install ../
fi

cmake --build . --target install --parallel

#Install pre-commit hook for formatting
cd $ROOT
cp Athos/HelperScripts/pre_commit_format_python.sh .git/hooks/pre-commit