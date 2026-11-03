import { SceneGraph } from '@open-pencil/scene-graph';
import { sceneGraphToDesignDocument, exportHTMLBundle } from '@open-pencil/dom-css';
export async function project(graph, opts = {}) {
  const doc = sceneGraphToDesignDocument(graph);
  const b = await exportHTMLBundle(doc, { html: 'standalone', style: opts.style ?? 'inline' });
  const e = b.files.find(f => f.path === b.entrypoint) ?? b.files[0];
  return String(e.content);
}
export { SceneGraph };
