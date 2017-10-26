import {Array1D, Array3D, Array4D, NDArray, NDArrayMathGPU, Scalar, Graph, Tensor, Session, SGDOptimizer, CostReduction, InCPUMemoryShuffledInputProviderBuilder} from 'deeplearn';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import * as math from './math';
import Game from './game';

const Target = Game.random('target');
const Current = Game.random('current');

const delay = (ms: number) => {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
};

ReactDOM.render(
    <div>
        <div style={{padding: "10px"}}>
            <h1>target</h1>
            {Target.canvas.render()}
        </div>
        <div style={{padding: "10px"}}>
            <h1>current</h1>
            {Current.canvas.render()}
        </div>
    </div>,
    document.getElementById("root")
);

Target.draw();
Current.draw();

const MAX_PIXEL = 255;

const imageDiff = (math: NDArrayMathGPU, T: NDArray, C: NDArray) => {
    const diff = math.sub(T, C).as1D();
    const outer = math.divide(math.dotProduct(diff, diff), Scalar.new(diff.size));
    return outer.as1D();
};

const product = (arr: number[]) => {
    let prod = 1;
    arr.forEach((a) => prod *= a);
    return prod;
};

class ConvANN {
    private graph: Graph;
    private session: Session;
    private input: Tensor;
    private target: Tensor;

    private prediction: Tensor;
    private cost: Tensor;

    private tmp: Tensor;

    buildConvLayer(prev: Tensor) {
        const id = Math.random();
        const convW = this.graph.variable('convW' + id, Array4D.randTruncatedNormal([5, 5, prev.shape[2], 32], 0, 0.1));
        const convB = this.graph.variable('convB' + id, Array1D.zeros([32]));
        const convLayer = this.graph.conv2d(prev, convW, convB, 5, 32);
        const reluLayer = this.graph.relu(convLayer);
        const maxPool = this.graph.maxPool(reluLayer, 2, 2);
        return reluLayer;
    }

    setup(inputs: number[], outputs: number[], math: NDArrayMathGPU) {
        this.graph = new Graph();
        this.input = this.graph.placeholder('input', inputs);
        this.target = this.graph.placeholder('target', outputs);

        const convLayer = this.buildConvLayer(this.input);
        const flat = this.graph.reshape(convLayer, [product(convLayer.shape)]);
        let layer1 = this.graph.layers.dense('layer1', flat, 128, (x) => this.graph.relu(x), true);
        let layer2 = this.graph.layers.dense('layer2', layer1, 64, (x) => this.graph.relu(x), true);
        let layer3 = this.graph.layers.dense('layer3', layer2, 32, (x) => this.graph.relu(x), true);
        let layer4 = this.graph.layers.dense('layer4', layer3, 16, (x) => this.graph.relu(x), true);
        let layer5 = this.graph.layers.dense('layer5', layer4, 1);
        this.prediction = layer5;
        this.cost = this.graph.meanSquaredCost(this.target, this.prediction);

        this.session = new Session(this.graph, math);
    }

    train(math: NDArrayMathGPU, image1s: Array3D[], image2s: Array3D[], ys: Array1D[]) {
        const optimizer = new SGDOptimizer(0.01);
        let cost;
        math.scope(() => {
            const stacked = image1s.map((i1, i) => math.concat3D(i1, image2s[i], 2));
            const providerBuilder = new InCPUMemoryShuffledInputProviderBuilder([
                stacked, ys
            ]);
            const [iprovider, yprovider] = providerBuilder.getInputProviders();
            const mapping = [{
                tensor: this.input,
                data: iprovider
            }, {
                tensor: this.target,
                data: yprovider
            }];

            const out = this.session.train(this.cost, mapping, 100, optimizer, CostReduction.MEAN);
            cost = out.get();
        });
        return cost;
    }

    predict(math: NDArrayMathGPU, image1: Array3D, image2: Array3D) {
        let pred;
        math.scope(() => {
            const stacked = math.concat3D(image1, image2, 2);
            const mapping = [{
                tensor: this.input,
                data: stacked
            }];

            pred = this.session.eval(this.prediction, mapping).getValues();
        });
        return pred;
    }
}

const ConvANN_merp = async () => {
    const math = new NDArrayMathGPU();
    const rawT = Target.getImage();
    const rawC = Current.getImage();

    const T = math.divide(Array3D.new(rawT.shape as [number, number, number], rawT.data, "float32"), Scalar.new(MAX_PIXEL)) as Array3D;
    const C = math.divide(Array3D.new(rawC.shape as [number, number, number], rawC.data, "float32"), Scalar.new(MAX_PIXEL)) as Array3D;

    const ann = new ConvANN();
    const [width, height, depth] = T.shape;
    ann.setup([width, height, depth * 2], [1], math);

    let i = 0;
    const run = async () => {
        const Ts = [];
        const Cs = [];
        const ys = [];

        for (let i = 0; i < 100000; ++i) {
            Target.clear();
            Target.randomize();
            Current.clear();
            Current.randomize();
            Target.draw();
            Current.draw();

            await delay(10);

            const rawT = Target.getImage();
            const rawC = Current.getImage();

            const T = math.divide(Array3D.new(rawT.shape as [number, number, number], rawT.data, "float32"), Scalar.new(MAX_PIXEL)) as Array3D;
            const C = math.divide(Array3D.new(rawC.shape as [number, number, number], rawC.data, "float32"), Scalar.new(MAX_PIXEL)) as Array3D;

            Ts.push(T);
            Cs.push(C);

            const y = imageDiff(math, T, C);

            ys.push(y);
        }
        console.log(i++, ann.train(math, Ts, Cs, ys));
    };
    run();
    setInterval(run, 45000);
};

ConvANN_merp();
