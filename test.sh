COMPRESSAI_PATH="$(python -c 'import compressai; print(compressai.__path__[0])')"
echo "$COMPRESSAI_PATH"
cd "$COMPRESSAI_PATH"
git rev-parse HEAD
