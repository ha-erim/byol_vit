import os
import csv
import matplotlib.pyplot as plt

def plot_accuracy_curve(log_path="logs", method="simsiam", save_path="accuracy_curve.png"):
    # CSV 로그 파일 경로 생성
    csv_file = os.path.join(log_path, f"acc_{method}.csv")
    if not os.path.exists(csv_file):
        print(f"[Error] 로그 파일이 존재하지 않습니다: {csv_file}")
        return

    # CSV에서 데이터를 읽어올 리스트 초기화
    epochs = []
    accs = []
    losses = []

    # CSV 파일 읽기
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # 첫 줄 헤더는 건너뜀
        for row in reader:
            if len(row) < 3:
                continue  # 빈 줄 또는 손상된 줄 건너뜀
            epochs.append(int(row[0]))         # Epoch
            accs.append(float(row[1]))         # Top-1 Accuracy
            losses.append(float(row[2]))       # Loss

    # 그래프 초기화 및 사이즈 설정
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # --------------------- Accuracy 곡선 ---------------------
    ax1.set_xlabel("Epoch")                          # x축: Epoch
    ax1.set_ylabel("Top-1 Acc(%)", color="tab:blue")  # 왼쪽 y축: Accuracy
    ax1.plot(epochs, accs, color="tab:blue", marker="o", label="acc")
    ax1.tick_params(axis="y", labelcolor="tab:blue")  # y축 색상 지정

    # --------------------- Loss 곡선 (오른쪽 y축) ---------------------
    ax2 = ax1.twinx()                                 # 같은 x축을 공유하는 두 번째 y축 생성
    ax2.set_ylabel("Loss", color="tab:green")           # 오른쪽 y축: Loss
    ax2.plot(epochs, losses, color="tab:green", marker="x", label="Loss")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # 그래프 제목 및 저장
    plt.title(f"Linear Probing (Acc & Loss)")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"[Saved] Accuracy + Loss curve 저장 완료: {save_path}")
    plt.close()  # 메모리 절약을 위해 그래프 닫기

if __name__ == "__main__":
    # --- 명령줄 인자 파싱 ---
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="byol",
                        help="학습 방식 (simsiam / byol / barlow)")
    parser.add_argument("--log_path", type=str, default="logs",
                        help="로그 파일이 저장된 디렉토리 경로")
    parser.add_argument("--save_path", type=str, default="accuracy_curve.png",
                        help="저장할 시각화 이미지 파일 경로")
    args = parser.parse_args()

    # 함수 실행
    plot_accuracy_curve(args.log_path, args.method, args.save_path)