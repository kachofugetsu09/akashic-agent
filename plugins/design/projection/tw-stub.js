export async function compile(){ throw new Error('tailwind 编译在浏览器侧不可用'); }
export async function compileAst(){ throw new Error('tailwind 不可用'); }
export default { compile, compileAst };
