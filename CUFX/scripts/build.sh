DIRECTORY="build"

if [ -d "$DIRECTORY" ]; then
    echo "";
    # rm -rf "$DIRECTORY"/*
else
    mkdir -p "$DIRECTORY"
fi

cd "$DIRECTORY"

# rm -r *

cmake ..

make -j8

