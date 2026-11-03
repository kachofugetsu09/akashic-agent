const nope = () => { throw new Error('node:fs 在浏览器侧不可用'); };
export const readFile = nope, writeFile = nope, mkdir = nope, readdir = nope, stat = nope, lstat = nope, unlink = nope, access = nope, copyFile = nope;
export default { readFile, writeFile, mkdir, readdir, stat, lstat, unlink, access, copyFile };
