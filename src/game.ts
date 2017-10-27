import Canvas from './components/canvas';

export const WIDTH = 400;
export const HEIGHT = 200;

export interface Point_t {
    x: number;
    y: number;
}

abstract class Shape {
    protected center: Point_t;
    protected canvas: Canvas;

    abstract draw(): any;
    abstract setCenter(point: Point_t): any;
    constructor (center: Point_t, canvas: Canvas) {
        this.center = center;
        this.canvas = canvas;
    }
}

class Rectangle extends Shape {
    private width: number;
    private height: number;

    draw(color = "#000000") {
        const point = this.center;
        const ctx = this.canvas.getContext();
        ctx.fillStyle=color;
        ctx.fillRect(point.x - (this.width / 2), point.y - (this.height / 2), this.width, this.height);
        ctx.stroke();
    }

    setCenter(point: Point_t) {
        this.center = point;
    }

    constructor (center: Point_t, canvas: Canvas, width = 40, height = 20) {
        super(center, canvas);
        this.width = width;
        this.height = height;
    }
}

export default class Game {
    private shapes: Shape[];
    private name: string;
    canvas: Canvas;

    static random(name: string) {
        const g = new Game(name);
        g.randomize();
        return g;
    }

    constructor (name: string) {
        this.name = name;
        this.shapes = [];
        this.canvas = new Canvas({
            width: WIDTH,
            height: HEIGHT,
            name
        });
    }

    randomize() {
        const rectangles = Math.ceil(Math.random() * 10);
        for (let i = 0; i < rectangles; ++i) {
            const randomCenter = this.randomCoords();
            this.placeRectangle(randomCenter);
        }
    }

    randomCoords() {
        return {
            x: Math.random() * WIDTH,
            y: Math.random() * HEIGHT
        };
    }

    placeRectangle(center: Point_t) {
        const r = new Rectangle(center, this.canvas);
        this.shapes.push(r);
    }

    undo() {
        this.shapes.pop();
    }

    draw() {
        this.shapes.forEach((shape) => shape.draw());
    }

    getImage() {
        return this.canvas.getImageData();
    }

    clear() {
        this.shapes = [];
        this.canvas.clear();
    }
}
