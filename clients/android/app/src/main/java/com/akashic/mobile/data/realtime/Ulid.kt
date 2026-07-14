package com.akashic.mobile.data.realtime

import java.security.SecureRandom

private const val CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

object Ulid {
    private val random = SecureRandom()

    fun next(nowMillis: Long = System.currentTimeMillis()): String {
        require(nowMillis in 0..0xffff_ffff_ffffL) { "ULID timestamp is out of range" }
        val bytes = ByteArray(16)
        for (index in 0 until 6) bytes[5 - index] = (nowMillis ushr (index * 8)).toByte()
        random.nextBytes(bytes, 6, 10)
        return encode(bytes)
    }

    private fun encode(bytes: ByteArray): String {
        var buffer = 0
        var bits = 2
        val result = StringBuilder(26)
        bytes.forEach { byte ->
            buffer = (buffer shl 8) or (byte.toInt() and 0xff)
            bits += 8
            while (bits >= 5) {
                bits -= 5
                result.append(CROCKFORD[(buffer ushr bits) and 31])
            }
        }
        return result.toString()
    }

    private fun SecureRandom.nextBytes(target: ByteArray, offset: Int, length: Int) {
        val randomPart = ByteArray(length)
        nextBytes(randomPart)
        randomPart.copyInto(target, offset)
    }
}
