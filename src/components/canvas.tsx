import * as React from 'react';

export default class Canvas {
    name: string;
    width: number;
    height: number;
    render() {
        return <canvas id={"canvas" + this.name} width={this.width} height={this.height} style={{"borderStyle": "solid"}}></canvas>
    }

    constructor(props: {name: string, width: number, height: number}) {
        this.name = props.name;
        this.width = props.width;
        this.height = props.height;
    }

    getContext() {
        const el = document.getElementById(`canvas${this.name}`) as any;
        return el.getContext("2d");
    }

    clear() {
        const ctx = this.getContext();
        ctx.clearRect(0, 0, this.width, this.height);
    }

    getImageData() {
        const ctx = this.getContext();
        const rawImageData = ctx.getImageData(0, 0, this.width, this.height).data as number[];
        const baw = [];
        for (let i = 0; i < rawImageData.length; i+=4) {
            baw.push(rawImageData[i+3]);
        }
        return {
            shape: [this.width, this.height, 1],
            data: baw
        };
    }
}
