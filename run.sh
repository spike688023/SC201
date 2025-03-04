param="${1:-update}"

echo "Commit msg is : $param"

git add -f **/*.py **/*.sh 2>/dev/null
git commit -m "$param"
git push -u origin main
