import json
import matplotlib.pyplot as plt


def get_scores(filename):
    scores = []

    with open(filename, "r") as file:
        for line in file:
            game = json.loads(line)
            scores.append(game["score"])

    return scores


random_scores = get_scores("results/random.jsonl")
greedy_scores = get_scores("results/greedy.jsonl")

agents = ["Random", "Greedy"]

average_scores = [
    sum(random_scores) / len(random_scores),
    sum(greedy_scores) / len(greedy_scores)
]

plt.bar(agents, average_scores)

plt.title("Random vs Greedy: Average Score")
plt.xlabel("AI Agent")
plt.ylabel("Average Score")

plt.savefig("results/score_comparison.png")
# Score distribution comparison
plt.figure()

plt.boxplot(
    [random_scores, greedy_scores],
    tick_labels=["Random", "Greedy"]
)

plt.title("Score Distribution: Random vs Greedy")
plt.xlabel("AI Agent")
plt.ylabel("Game Score")
plt.grid(axis="y", alpha=0.3)

plt.savefig("results/score_distribution.png")


# GRAPH 3: Maximum tile comparison

def get_max_tiles(filename):
    max_tiles = []

    with open(filename, "r") as file:
        for line in file:
            game = json.loads(line)
            max_tiles.append(game["max_tile"])

    return max_tiles


random_tiles = get_max_tiles("results/random.jsonl")
greedy_tiles = get_max_tiles("results/greedy.jsonl")

plt.figure()
plt.hist(
    [random_tiles, greedy_tiles],
    label=["Random", "Greedy"],
    bins=[0, 64, 128, 256, 512, 1024, 2048, 4096]
)

plt.title("Maximum Tile Distribution")
plt.xlabel("Maximum Tile Reached")
plt.ylabel("Number of Games")
plt.legend()
plt.savefig("results/max_tile_comparison.png")
plt.show()