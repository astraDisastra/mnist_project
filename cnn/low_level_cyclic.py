import html
import math
import os
import random
import webbrowser



def dot(a, b):
    """Dot product of two vectors."""
    total = 0.0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def matvec(matrix, vector):
    """Matrix-vector multiplication."""
    output = []

    for row in matrix:
        output.append(dot(row, vector))

    return output


def vector_add(a, b):
    """Add two vectors."""
    output = []

    for i in range(len(a)):
        output.append(a[i] + b[i])

    return output


def tanh_vector(vector):
    """Apply tanh to every value in a vector."""
    output = []

    for value in vector:
        output.append(math.tanh(value))

    return output


def sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def binary_cross_entropy(prediction, target):
    """Loss for binary classification."""
    epsilon = 1e-9

    prediction = max(epsilon, min(1.0 - epsilon, prediction))

    return -(
        target * math.log(prediction)
        + (1 - target) * math.log(1 - prediction)
    )



class CyclicNeuralNetwork:
    def __init__(
        self,
        input_dim=2,
        hidden_dim=6,
        num_nodes=3,
        steps=3,
        seed=1,
    ):
        random.seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.steps = steps

        self.incoming = []

        for i in range(num_nodes):
            previous_node = (i - 1) % num_nodes
            self.incoming.append(previous_node)

        self.node_weights = []
        self.node_biases = []

        for _ in range(num_nodes):
            node_input_dim = input_dim + hidden_dim

            weights = []

            for _ in range(hidden_dim):
                row = []

                for _ in range(node_input_dim):
                    row.append(random.uniform(-0.5, 0.5))

                weights.append(row)

            bias = []

            for _ in range(hidden_dim):
                bias.append(0.0)

            self.node_weights.append(weights)
            self.node_biases.append(bias)

        readout_input_dim = num_nodes * hidden_dim

        self.output_weights = []

        for _ in range(readout_input_dim):
            self.output_weights.append(random.uniform(-0.5, 0.5))

        self.output_bias = [0.0]

    def forward(self, x):
        
        states = []

        for _ in range(self.num_nodes):
            states.append([0.0 for _ in range(self.hidden_dim)])

        # Propagate information around the cycle several times.
        for _ in range(self.steps):
            old_states = states
            new_states = []

            for node_index in range(self.num_nodes):
                incoming_node = self.incoming[node_index]

                neighbor_state = old_states[incoming_node]

                node_input = x + neighbor_state

                z = matvec(
                    self.node_weights[node_index],
                    node_input,
                )

                z = vector_add(
                    z,
                    self.node_biases[node_index],
                )

                h = tanh_vector(z)

                new_states.append(h)

            states = new_states

        graph_representation = []

        for state in states:
            for value in state:
                graph_representation.append(value)

        # Final prediction.
        logit = dot(self.output_weights, graph_representation)
        logit += self.output_bias[0]

        prediction = sigmoid(logit)

        return prediction

    def loss(self, x, y):
        prediction = self.forward(x)
        return binary_cross_entropy(prediction, y)

    def parameters(self):

        for weights in self.node_weights:
            for row in weights:
                for j in range(len(row)):
                    yield row, j

        for bias in self.node_biases:
            for j in range(len(bias)):
                yield bias, j

        for j in range(len(self.output_weights)):
            yield self.output_weights, j

        yield self.output_bias, 0


def parameter_snapshot(model):
    """Return stable parameter labels and values for visualization."""
    labels = []
    values = []

    for node_index in range(model.num_nodes):
        incoming_node = model.incoming[node_index]
        input_labels = []

        for input_index in range(model.input_dim):
            input_labels.append(f"x{input_index}")

        for hidden_index in range(model.hidden_dim):
            input_labels.append(f"N{incoming_node}.h{hidden_index}")

        for hidden_index in range(model.hidden_dim):
            row = model.node_weights[node_index][hidden_index]

            for input_index, value in enumerate(row):
                labels.append(
                    f"N{node_index}.W[h{hidden_index}<-{input_labels[input_index]}]"
                )
                values.append(value)

        for hidden_index, value in enumerate(model.node_biases[node_index]):
            labels.append(f"N{node_index}.b[h{hidden_index}]")
            values.append(value)

    weight_index = 0

    for node_index in range(model.num_nodes):
        for hidden_index in range(model.hidden_dim):
            labels.append(f"Out.W[N{node_index}.h{hidden_index}]")
            values.append(model.output_weights[weight_index])
            weight_index += 1

    labels.append("Out.b")
    values.append(model.output_bias[0])

    return labels, values


def capture_training_state(model, data, epoch):
    labels, values = parameter_snapshot(model)

    return {
        "epoch": epoch,
        "loss": average_loss(model, data),
        "accuracy": accuracy(model, data),
        "labels": labels,
        "values": values,
    }


def color_for_value(value, max_abs):
    if max_abs == 0:
        intensity = 0
    else:
        intensity = min(1.0, abs(value) / max_abs)

    channel = int(255 - 180 * intensity)

    if value >= 0:
        return f"rgb({channel},{channel},255)"

    return f"rgb(255,{channel},{channel})"


def render_network_graph(model):
    width = 900
    height = 420
    center_x = 450
    center_y = 210
    radius = 120
    node_positions = []

    for node_index in range(model.num_nodes):
        angle = -math.pi / 2 + (2 * math.pi * node_index / model.num_nodes)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        node_positions.append((x, y))

    input_positions = []

    for input_index in range(model.input_dim):
        if model.input_dim == 1:
            y = center_y
        else:
            y = 110 + input_index * 200 / (model.input_dim - 1)

        input_positions.append((90, y))

    output_position = (810, 210)

    parts = [
        '<svg class="network" viewBox="0 0 900 420" role="img" '
        'aria-label="Cyclic neural network graph">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#273043" />',
        "</marker>",
        "</defs>",
    ]

    for input_index, (x, y) in enumerate(input_positions):
        parts.append(f'<circle class="input-node" cx="{x}" cy="{y}" r="30" />')
        parts.append(
            f'<text class="node-label" x="{x}" y="{y + 5}">x{input_index}</text>'
        )

    for input_index, (input_x, input_y) in enumerate(input_positions):
        for node_x, node_y in node_positions:
            parts.append(
                '<line class="edge muted" '
                f'x1="{input_x + 32}" y1="{input_y}" '
                f'x2="{node_x - 42}" y2="{node_y}" marker-end="url(#arrow)" />'
            )

    for node_index, incoming_index in enumerate(model.incoming):
        start_x, start_y = node_positions[incoming_index]
        end_x, end_y = node_positions[node_index]
        parts.append(
            '<path class="edge strong" '
            f'd="M {start_x} {start_y} Q {center_x} {center_y} {end_x} {end_y}" '
            'marker-end="url(#arrow)" />'
        )

    for node_index, (x, y) in enumerate(node_positions):
        parts.append(f'<circle class="cycle-node" cx="{x}" cy="{y}" r="44" />')
        parts.append(f'<text class="node-label" x="{x}" y="{y - 4}">N{node_index}</text>')
        parts.append(
            f'<text class="node-subtitle" x="{x}" y="{y + 18}">'
            f'{model.hidden_dim} hidden</text>'
        )

    for node_index, (node_x, node_y) in enumerate(node_positions):
        parts.append(
            '<line class="edge readout" '
            f'x1="{node_x + 45}" y1="{node_y}" '
            f'x2="{output_position[0] - 38}" y2="{output_position[1]}" '
            'marker-end="url(#arrow)" />'
        )

    x, y = output_position
    parts.append(f'<circle class="output-node" cx="{x}" cy="{y}" r="38" />')
    parts.append(f'<text class="node-label" x="{x}" y="{y + 5}">Out</text>')
    parts.append(
        '<text class="graph-caption" x="450" y="392">'
        'Each cyclic node receives x plus the previous node state, then the readout '
        'uses all final node states.</text>'
    )
    parts.append("</svg>")

    return "\n".join(parts)


def render_line_chart(history, key, title, stroke, y_min=None, y_max=None):
    width = 820
    height = 220
    left = 54
    right = 18
    top = 24
    bottom = 38
    values = [item[key] for item in history]

    if y_min is None:
        y_min = min(values)

    if y_max is None:
        y_max = max(values)

    if y_max == y_min:
        y_max = y_min + 1.0

    points = []

    for index, value in enumerate(values):
        if len(values) == 1:
            x = left
        else:
            x = left + index * (width - left - right) / (len(values) - 1)

        y = top + (y_max - value) * (height - top - bottom) / (y_max - y_min)
        points.append(f"{x:.1f},{y:.1f}")

    last_value = values[-1]

    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <text class="chart-title" x="{left}" y="16">{html.escape(title)}</text>
  <line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" />
  <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" />
  <text class="tick" x="8" y="{top + 5}">{y_max:.3f}</text>
  <text class="tick" x="8" y="{height - bottom + 4}">{y_min:.3f}</text>
  <polyline class="series" style="stroke:{stroke}" points="{' '.join(points)}" />
  <text class="last-value" x="{width - right - 155}" y="{top + 18}">final {last_value:.4f}</text>
</svg>
"""


def render_parameter_heatmap(history):
    labels = history[-1]["labels"]
    epoch_count = len(history)
    max_abs = 0.0

    for item in history:
        for value in item["values"]:
            max_abs = max(max_abs, abs(value))

    cell_width = 4
    row_height = 12
    label_width = 210
    top = 34
    width = label_width + epoch_count * cell_width + 24
    height = top + len(labels) * row_height + 42

    parts = [
        f'<svg class="heatmap" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Weights and biases across training epochs">',
        f'<text class="chart-title" x="{label_width}" y="18">'
        'Weights and biases by epoch</text>',
    ]

    for row_index, label in enumerate(labels):
        y = top + row_index * row_height
        parts.append(
            f'<text class="param-label" x="4" y="{y + 9}">'
            f'{html.escape(label)}</text>'
        )

        for column_index, item in enumerate(history):
            value = item["values"][row_index]
            x = label_width + column_index * cell_width
            color = color_for_value(value, max_abs)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{row_height}" '
                f'fill="{color}"><title>{html.escape(label)} | '
                f'epoch {item["epoch"]}: {value:.6f}</title></rect>'
            )

    legend_y = height - 28
    parts.append(f'<text class="legend" x="{label_width}" y="{legend_y}">negative</text>')
    parts.append(
        f'<rect x="{label_width + 62}" y="{legend_y - 10}" width="60" height="10" '
        'fill="rgb(255,75,75)" />'
    )
    parts.append(
        f'<rect x="{label_width + 126}" y="{legend_y - 10}" width="60" height="10" '
        'fill="rgb(75,75,255)" />'
    )
    parts.append(f'<text class="legend" x="{label_width + 194}" y="{legend_y}">positive</text>')
    parts.append(
        f'<text class="legend" x="{label_width}" y="{legend_y + 16}">'
        f'Showing all {epoch_count} epochs; hover cells for exact values.'
        '</text>'
    )
    parts.append("</svg>")

    return "\n".join(parts)


def render_final_parameter_tables(model):
    rows = []

    for node_index in range(model.num_nodes):
        for hidden_index, row in enumerate(model.node_weights[node_index]):
            weights = ", ".join(f"{value:.4f}" for value in row)
            bias = model.node_biases[node_index][hidden_index]
            rows.append(
                "<tr>"
                f"<td>N{node_index}.h{hidden_index}</td>"
                f"<td>{html.escape(weights)}</td>"
                f"<td>{bias:.4f}</td>"
                "</tr>"
            )

    output_weights = ", ".join(f"{value:.4f}" for value in model.output_weights)

    return f"""
<table>
  <thead>
    <tr><th>Unit</th><th>Weights</th><th>Bias</th></tr>
  </thead>
  <tbody>
    {''.join(rows)}
    <tr><td>Output</td><td>{html.escape(output_weights)}</td><td>{model.output_bias[0]:.4f}</td></tr>
  </tbody>
</table>
"""


def write_training_report(model, history, output_path):
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cyclic neural network training report</title>
  <style>
    body {{
      margin: 0;
      background: #f7f4ed;
      color: #273043;
      font-family: Arial, Helvetica, sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1, h2 {{
      margin: 0 0 14px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 32px;
    }}
    h2 {{
      margin-top: 34px;
      font-size: 22px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0 24px;
    }}
    .metric {{
      background: #fffdf8;
      border: 1px solid #ddd6c7;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric span {{
      display: block;
      color: #726b60;
      font-size: 13px;
      margin-bottom: 5px;
    }}
    .metric strong {{
      font-size: 24px;
    }}
    .panel {{
      overflow-x: auto;
      background: #fffdf8;
      border: 1px solid #ddd6c7;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 18px;
    }}
    svg {{
      display: block;
      max-width: 100%;
      height: auto;
    }}
    .network {{
      min-width: 820px;
    }}
    .input-node {{
      fill: #f2c94c;
      stroke: #273043;
      stroke-width: 2;
    }}
    .cycle-node {{
      fill: #9ccfc8;
      stroke: #273043;
      stroke-width: 2.5;
    }}
    .output-node {{
      fill: #f2994a;
      stroke: #273043;
      stroke-width: 2.5;
    }}
    .edge {{
      fill: none;
      stroke: #273043;
      stroke-width: 2;
    }}
    .edge.muted {{
      stroke: #a69d8e;
      stroke-width: 1.4;
      opacity: 0.55;
    }}
    .edge.strong {{
      stroke-width: 3;
    }}
    .edge.readout {{
      stroke: #5f6f94;
      opacity: 0.75;
    }}
    .node-label, .node-subtitle, .graph-caption, .chart-title, .tick,
    .last-value, .param-label, .legend {{
      text-anchor: middle;
      font-family: Arial, Helvetica, sans-serif;
      fill: #273043;
    }}
    .node-label {{
      font-size: 18px;
      font-weight: 700;
    }}
    .node-subtitle {{
      font-size: 11px;
    }}
    .graph-caption {{
      font-size: 13px;
    }}
    .chart {{
      min-width: 720px;
    }}
    .axis {{
      stroke: #8d867a;
      stroke-width: 1;
    }}
    .series {{
      fill: none;
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .chart-title {{
      text-anchor: start;
      font-size: 14px;
      font-weight: 700;
    }}
    .tick, .last-value, .legend {{
      text-anchor: start;
      font-size: 11px;
      fill: #5d574d;
    }}
    .heatmap {{
      max-width: none;
    }}
    .param-label {{
      text-anchor: start;
      font-size: 9px;
      fill: #3c352e;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #ddd6c7;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f2eadc;
    }}
    @media (max-width: 760px) {{
      main {{
        padding: 22px 12px 40px;
      }}
      .summary {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Cyclic neural network training report</h1>
    <div class="summary">
      <div class="metric"><span>Epochs</span><strong>{len(history)}</strong></div>
      <div class="metric"><span>Final loss</span><strong>{history[-1]["loss"]:.4f}</strong></div>
      <div class="metric"><span>Final accuracy</span><strong>{history[-1]["accuracy"]:.2f}</strong></div>
    </div>

    <h2>Network graph</h2>
    <div class="panel">{render_network_graph(model)}</div>

    <h2>Training progress</h2>
    <div class="panel">{render_line_chart(history, "loss", "Loss by epoch", "#d1495b")}</div>
    <div class="panel">{render_line_chart(history, "accuracy", "Accuracy by epoch", "#24746b", 0.0, 1.0)}</div>

    <h2>Weights and biases through each epoch</h2>
    <div class="panel">{render_parameter_heatmap(history)}</div>

    <h2>Final weights and biases</h2>
    <div class="panel">{render_final_parameter_tables(model)}</div>
  </main>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(document)



def numerical_gradient_update(model, x, y, learning_rate=0.2, epsilon=1e-4):


    updates = []

    for container, index in model.parameters():
        original_value = container[index]

        container[index] = original_value + epsilon
        loss_plus = model.loss(x, y)

        container[index] = original_value - epsilon
        loss_minus = model.loss(x, y)

        gradient = (loss_plus - loss_minus) / (2 * epsilon)

        container[index] = original_value

        updates.append((container, index, gradient))

    for container, index, gradient in updates:
        container[index] -= learning_rate * gradient


def average_loss(model, data):
    total = 0.0

    for x, y in data:
        total += model.loss(x, y)

    return total / len(data)


def accuracy(model, data):
    correct = 0

    for x, y in data:
        prediction = model.forward(x)

        if prediction >= 0.5:
            predicted_class = 1
        else:
            predicted_class = 0

        if predicted_class == y:
            correct += 1

    return correct / len(data)


def train(model, data, epochs=1000, learning_rate=0.2):
    history = []

    for epoch in range(epochs):
        shuffled_data = data[:]
        random.shuffle(shuffled_data)

        for x, y in shuffled_data:
            numerical_gradient_update(
                model,
                x,
                y,
                learning_rate=learning_rate,
            )

        history.append(capture_training_state(model, data, epoch + 1))

        if (epoch + 1) % 50 == 0:
            loss = history[-1]["loss"]
            acc = history[-1]["accuracy"]

            print(f"Epoch {epoch + 1}")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {acc:.2f}")

            for x, y in data:
                prediction = model.forward(x)
                print(f"  input={x}, target={y}, prediction={prediction:.4f}")

            print()

    return history


# -----------------------------
# Main program
# -----------------------------

def main():

    data = [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 0),
    ]

    model = CyclicNeuralNetwork(
        input_dim=2,
        hidden_dim=4,
        num_nodes=3,
        steps=3,
        seed=1,
    )

    print("Before training:")

    for x, y in data:
        prediction = model.forward(x)
        print(f"  input={x}, target={y}, prediction={prediction:.4f}")

    print()

    history = train(
        model,
        data,
        epochs=1000,
        learning_rate=0.2,
    )

    print("After training:")

    for x, y in data:
        prediction = model.forward(x)
        predicted_class = 1 if prediction >= 0.5 else 0

        print(
            f"  input={x}, "
            f"target={y}, "
            f"prediction={prediction:.4f}, "
            f"class={predicted_class}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, "cyclic_training_report.html")
    write_training_report(model, history, report_path)

    print()
    print(f"Training visualization written to: {report_path}")

    try:
        webbrowser.open(f"file://{report_path}")
    except webbrowser.Error:
        pass


if __name__ == "__main__":
    main()
