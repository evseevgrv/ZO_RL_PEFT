#!/bin/bash

# Deep Optuna Hyperparameter Optimization
# This script runs extensive hyperparameter search with WandB logging

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Optuna Deep Hyperparameter Optimization - ZO-RL-Jaguar       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Environment Configuration
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="false"
# export WANDB_ENTITY="andrey"
export WANDB_API_KEY=""  # Вставьте ваш WandB API key здесь
export HF_TOKEN=""  # Если нужен

# Optuna Configuration
N_TRIALS=50  # Количество экспериментов для глубокого исследования
STUDY_NAME="zo_rl_jaguar_deep_$(date +%Y%m%d_%H%M%S)"
STORAGE="sqlite:///optuna_deep.db"

echo "Настройки:"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  WandB Project: zo-rl-jaguar-optuna"
echo "  WandB Entity: $WANDB_ENTITY"
echo "  Количество trials: $N_TRIALS"
echo "  Study name: $STUDY_NAME"
echo "  Storage: $STORAGE"
echo ""

# Check if continuing existing study
if [ -f "optuna_deep.db" ]; then
    echo "⚠️  Найдена существующая база данных optuna_deep.db"
    read -p "Продолжить существующее исследование? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        LOAD_FLAG="--load_if_exists"
        echo "✓ Продолжаем существующее исследование..."
    else
        echo "✓ Создаем новое исследование..."
        LOAD_FLAG=""
    fi
else
    LOAD_FLAG=""
    echo "✓ Создаем новое исследование..."
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Запуск оптимизации..."
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Примерное время: ~$(echo "$N_TRIALS * 7.5" | bc) часов (~$(echo "$N_TRIALS * 7.5 / 24" | bc) дней)"
echo ""
echo "Для остановки: Ctrl+C (прогресс сохранится в БД)"
echo "Для продолжения: запустите этот скрипт снова"
echo ""

# Run Optuna optimization
python optuna_advanced.py \
    --n_trials $N_TRIALS \
    --study_name "$STUDY_NAME" \
    --storage "$STORAGE" \
    $LOAD_FLAG \
    --lr_min 1e-5 \
    --lr_max 1e-1 \
    --lr_mu_min 1e-5 \
    --lr_mu_max 1e-1

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Оптимизация завершена успешно!"
    echo ""
    echo "Результаты сохранены в:"
    echo "  - result/optuna_best_params_${STUDY_NAME}.json"
    echo "  - $STORAGE"
    echo ""
    echo "Для анализа результатов:"
    echo "  python -c \"import optuna; study = optuna.load_study(study_name='$STUDY_NAME', storage='$STORAGE'); print('Best value:', study.best_value); print('Best params:', study.best_params)\""
    echo ""
    echo "Лучшие параметры:"
    if [ -f "result/optuna_best_params_${STUDY_NAME}.json" ]; then
        cat "result/optuna_best_params_${STUDY_NAME}.json" | python -m json.tool 2>/dev/null || cat "result/optuna_best_params_${STUDY_NAME}.json"
    fi
else
    echo "⚠️  Оптимизация была прервана или завершилась с ошибкой"
    echo ""
    echo "Для продолжения запустите скрипт снова:"
    echo "  ./run_optuna_deep.sh"
fi

echo "════════════════════════════════════════════════════════════════"












