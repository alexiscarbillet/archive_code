const timesTwo = require('./timesTwo');

describe('timesTwo function', () => {
  test('returns the number times 2', () => {
    expect(timesTwo(10)).toBe(20);
  });
});