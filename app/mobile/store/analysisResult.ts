import { AnalyzeResponse } from "../api/types";

export class AnalysisResultStore {
  private static _latest: AnalyzeResponse | null = null;
  private static _session: AnalyzeResponse[] = [];

  static set(result: AnalyzeResponse): void {
    this._latest = result;
    this._session = [result];
  }

  static setSession(results: AnalyzeResponse[]): void {
    this._session = results;
    this._latest = results[results.length - 1] ?? null;
  }

  static get(): AnalyzeResponse | null {
    return this._latest;
  }

  static getSession(): AnalyzeResponse[] {
    return this._session;
  }
}
