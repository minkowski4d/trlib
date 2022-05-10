#source $(dirname "${BASH_SOURCE[0]}")/checkout.sh

version=$1
if [ "$version" == "" ]; then
  version="latest"
fi

docker build -t devhub.intra.quantumrock.de:5005/python/portfolio-dashboard:$version .