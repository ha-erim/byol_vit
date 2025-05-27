import os
import csv
import matplotlib.pyplot as plt

def plot_loss_curve(log_path, method="simsiam", save_path="loss_curve.png"):
    """
    학습 중 평균 loss 로그를 시각화하여 PNG 파일로 저장합니다.

    Args:
        log_path (str): CSV 로그 파일이 위치한 디렉토리 (예: logs/)
        method (str): 학습 방식 이름 (파일명에 사용됨)
        save_path (str): 그래프를 저장할 경로 및 파일 이름
    """

    # 로그 CSV 파일 경로 구성
    csv_file = os.path.join(log_path, f"loss_{method}.csv")
    if not os.path.exists(csv_file):
        print(f"[Error] 로그 파일이 존재하지 않습니다: {csv_file}")
        return

    # 로그 값을 담을 리스트 초기화
    epochs = []
    losses = []

    # CSV 파일 열기 및 내용 파싱
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더(예: epoch,avg_loss)는 건너뜀
        for row in reader:
            if len(row) < 2:
                continue  # 빈 줄 혹은 손상된 줄 방지
            epochs.append(int(row[0]))         # Epoch 번호
            losses.append(float(row[1]))       # 평균 Loss 값

    # 시각화: Loss 곡선 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', label=f"{method} Loss")  # 점선 그래프
    plt.xlabel("Epoch")             # x축: 학습 Epoch
    plt.ylabel("Average Loss")      # y축: Loss 값
    plt.title(f"Training Loss")  # 그래프 제목
    plt.grid(True)
    plt.legend()

    # 그래프 저장
    plt.savefig(save_path)
    print(f"[Saved] Loss curve 저장 완료: {save_path}")
    plt.close()  # 리소스 정리

if __name__ == "__main__":
    # --- 명령줄 인자 처리 ---
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="byol",
                        help="학습 방식 (예: simsiam / byol / barlow)")
    parser.add_argument("--log_path", type=str, default="logs",
                        help="로그 파일이 위치한 폴더 경로")
    parser.add_argument("--save_path", type=str, default="loss_curve.png",
                        help="그래프 저장 경로 (파일명 포함)")
    args = parser.parse_args()

    # 시각화 함수 실행
    plot_loss_curve(args.log_path, args.method, args.save_path)