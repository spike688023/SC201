param="${1:-update}"

echo "Commit msg is : $param"

git add *.py *.sh
git commit -m $param
git push -u origin main
