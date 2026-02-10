CONFIG_DIR="./config"

python main.py --config "$CONFIG_DIR/main_framework.yaml"

if [ $? -eq 0 ]; then
    echo "✅ Finished successfully"
else
    echo "❌ Failed"
    exit 1
fi

echo ""