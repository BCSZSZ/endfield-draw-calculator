import { plan } from './planner.mjs';

self.onmessage = ({ data }) => {
  try {
    self.postMessage({ result: plan(data.first, data.second, data.config) });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
