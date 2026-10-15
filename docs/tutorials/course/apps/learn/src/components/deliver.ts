import { DeliveryLab as Engine } from "engine";
import type { Delivery, DeliveryCase } from "engine";

/** One term, five deliveries — the modeled column is live. */
export class Deliver {
  private d = new Engine();

  get cases(): readonly DeliveryCase[] {
    return this.d.cases;
  }

  get sel(): number {
    return this.d.sel;
  }

  get deliveryCase(): DeliveryCase {
    return this.d.deliveryCase;
  }

  get picked(): number {
    return this.d.picked;
  }

  get delivery(): Delivery {
    return this.d.delivery;
  }

  get rows(): { delivery: Delivery; modeledUs: number; modelPick: boolean; winner: boolean }[] {
    return this.d.rows;
  }

  get modelPick(): Delivery {
    return this.d.modelPick;
  }

  get winner(): Delivery {
    return this.d.winner;
  }

  get agrees(): boolean {
    return this.d.agrees;
  }

  get speedup(): number {
    return this.d.speedup;
  }

  select(i: number): void {
    this.d.select(i);
  }

  choose(k: number): void {
    this.d.choose(k);
  }
}
