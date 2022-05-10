rm -rf build/*
while read p || [ -n "$p" ]; do
  arr=(${p//=/ })
  LIB="${arr[0]}"
  TAG="${arr[1]}"
  REPO="git@devhub.intra.quantumrock.de:$LIB.git"
  echo "Checking out $REPO $TAG"
  git clone -b $TAG $REPO build/$(basename $LIB)
done < ./git-dependencies.txt