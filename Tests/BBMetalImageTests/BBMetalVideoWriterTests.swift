import AVFoundation
import Metal
import XCTest
@testable import BBMetalImage

final class BBMetalVideoWriterTests: XCTestCase {
    func testAudioSettingsRejectInvalidSampleRates() {
        let invalidSampleRates: [Double] = [.nan, .infinity, -.infinity, -1, 0, 7_999, 192_001]
        for sampleRate in invalidSampleRates {
            XCTAssertThrowsError(try BBMetalVideoWriter.audioOutputSettings(sampleRate: sampleRate), "Invalid sample rate: \(sampleRate)") { error in
                guard case BBMetalVideoWriterError.invalidAudioSampleRate = error else {
                    return XCTFail("Unexpected sample rate validation error: \(error)")
                }
            }
        }
    }

    func testAudioSettingsPreserveFiniteSampleRatesWithinBounds() throws {
        for sampleRate: Double in [8_000, 44_100, 48_000, 192_000] {
            let settings = try BBMetalVideoWriter.audioOutputSettings(sampleRate: sampleRate)
            XCTAssertEqual(settings[AVSampleRateKey] as? Double, sampleRate)
            XCTAssertEqual(settings[AVNumberOfChannelsKey] as? Int, 1)
            XCTAssertEqual(settings[AVFormatIDKey] as? UInt32, kAudioFormatMPEG4AAC)
        }
    }

    func testDuplicateStartPreservesActiveOutputURL() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let originalURL = directory.appendingPathComponent("original.mp4")
        let writer = BBMetalVideoWriter(frameSize: BBMetalIntSize(width: 16, height: 16))
        writer.hasAudioTrack = false
        var originalStarts = 0
        var duplicateStarts = 0
        XCTAssertNil(writer.start(url: originalURL, startHandler: { _ in originalStarts += 1 }))
        defer { writer.cancel() }
        let error = writer.start(url: directory.appendingPathComponent("duplicate.mp4"),
                                 startHandler: { _ in duplicateStarts += 1 })
        guard let error, case BBMetalVideoWriterError.startCalledBeforePreviousWriteFinished = error else {
            return XCTFail("Duplicate start must fail")
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 16, height: 16, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        let texture = try XCTUnwrap(BBMetalDevice.sharedDevice.makeTexture(descriptor: descriptor))
        let source = BBMetalStaticImageSource(texture: texture)
        writer.newTextureAvailable(BBMetalDefaultTexture(metalTexture: texture, sampleTime: .zero), from: source)
        XCTAssertEqual(originalStarts, 1)
        XCTAssertEqual(duplicateStarts, 0)
        let finished = expectation(description: "Original recording finishes at its own URL")
        writer.finish { result in
            switch result {
            case let .success(url): XCTAssertEqual(url, originalURL)
            case let .failure(error): XCTFail("Recording failed: \(error)")
            }
            finished.fulfill()
        }
        wait(for: [finished], timeout: 5)
    }

    func testStartReportsExistingFileFailureAndAllowsRetry() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let existingURL = directory.appendingPathComponent("existing.mp4")
        try Data([0]).write(to: existingURL)

        let writer = BBMetalVideoWriter(frameSize: BBMetalIntSize(width: 16, height: 16))
        writer.hasAudioTrack = false

        XCTAssertNotNil(writer.start(url: existingURL), "An existing output file must not be reported as a successful start")
        XCTAssertNotNil(writer.start(url: existingURL), "A failed start must allow another attempt")
        XCTAssertNil(writer.start(url: directory.appendingPathComponent("retry.mp4")))
        writer.cancel()
    }
}
